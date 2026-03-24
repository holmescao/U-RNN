"""
flood_head.py — Dual-output prediction head for U-RNN.

Implements a YOLOX-style decoupled head with two independent branches:

  cls branch: wet/dry flood extent classification
    Conv→Conv→sigmoid → probability map ∈ [0, 1] (B, 1, H, W)

  reg branch: flood depth regression
    Conv→Conv→LeakyReLU → depth map ≥ 0 (B, 1, H, W) in normalised units

The classification output serves as a spatial mask on the regression output:
cells predicted dry (cls probability < cls_thred) have their depth zeroed out
(``correction_depth``). This prevents false positive depth values in building
grid cells and at domain edges — a key advantage over a single regression head.

During training, both branches contribute to ``FocalBCE_and_WMSE`` via the
``{"reg": ..., "cls": ...}`` dict assembled in ``main.accumulate_predictions``.
"""

import torch
import torch.nn as nn
from .network_blocks import BaseConv, DWConv, finalConv
from torch.utils.checkpoint import checkpoint


class ModuleWrapperIgnores2ndArg_cnn(nn.Module):
    """Checkpoint-compatible wrapper that ignores a dummy second argument."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None, "dummy_arg is required but was None"
        return self.module(x)


class YOLOXHead(nn.Module):
    """Decoupled prediction head producing flood depth and wet/dry probability.

    Inspired by the YOLOX decoupled head (Ge et al. 2021), the two branches
    share a stem layer but diverge into independent convolutional sub-networks.
    This allows the classification and regression tasks to develop specialised
    feature representations without interfering with each other.

    Parameters
    ----------
    cls_thred : float
        Wet/dry probability threshold for ``correction_depth`` (default 0.5).
        Cells with cls probability < cls_thred have predicted depth set to 0.
    in_channels : int
        Input feature channels from the decoder (default 64).
    width : float
        Channel width multiplier (default 0.25 → 16 channels per branch).
    depthwise : bool
        Use depthwise-separable convolutions (reduces parameters).
    use_checkpoint : bool
        Enable gradient checkpointing.
    """

    def __init__(
        self,
        cls_thred=0.5,
        in_channels=64,
        width=0.25,
        depthwise=False,
        use_checkpoint=True,
        input_height=500,
        input_width=500,
    ):
        super().__init__()

        Conv = DWConv if depthwise else BaseConv

        # Activation schedule: stem→cls_conv→reg_conv→cls_pred→reg_pred
        acts = ["silu"] * 3 + ["sigmoid", "lrelu"]
        self.acts = acts

        ch = int(in_channels * width)
        H, W = input_height, input_width

        # Shared stem: reduces input features before branching
        self.stems = BaseConv(
            in_channels=ch, out_channels=ch,
            ksize=1, stride=1, act=acts[0],
            height=H, width=W,
        )

        # Classification branch: two convolutions → wet/dry probability
        self.cls_convs = nn.Sequential(*[
            Conv(in_channels=ch, out_channels=ch,
                 ksize=1, stride=1, act=acts[1],
                 height=H, width=W),
            Conv(in_channels=ch, out_channels=ch,
                 ksize=1, stride=1, act=acts[1],
                 height=H, width=W),
        ])

        # Regression branch: two convolutions → flood depth
        self.reg_convs = nn.Sequential(*[
            Conv(in_channels=ch, out_channels=ch,
                 ksize=1, stride=1, act=acts[2],
                 height=H, width=W),
            Conv(in_channels=ch, out_channels=ch,
                 ksize=1, stride=1, act=acts[2],
                 height=H, width=W),
        ])

        # 1×1 final projections: features → single-channel output maps
        self.cls_preds = finalConv(  # sigmoid output in [0,1]
            in_channels=ch, out_channels=1,
            ksize=1, stride=1, act=acts[3], norm="",
        )
        self.reg_preds = finalConv(  # leaky-ReLU output ≥ 0
            in_channels=ch, out_channels=1,
            ksize=1, stride=1, act=acts[4], norm="",
        )

        # Checkpoint wrappers for gradient-memory trade-off
        self.stems_wrapper    = ModuleWrapperIgnores2ndArg_cnn(self.stems)
        self.cls_convs_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.cls_convs)
        self.reg_convs_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.reg_convs)
        self.cls_preds_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.cls_preds)
        self.reg_preds_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.reg_preds)

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.use_checkpoint = use_checkpoint
        self.cls_thred = cls_thred

    def forward(self, inputs):
        """Apply the dual-output head to a batch of spatiotemporal features.

        Parameters
        ----------
        inputs : Tensor, shape (S, B, F, H, W)
            Decoder output feature maps (S timesteps, B batch, F features).

        Returns
        -------
        Tensor, shape (S, B, 2, H, W)
            Channel 0: flood depth (normalised, dry cells zeroed out)
            Channel 1: wet/dry probability ∈ [0, 1]
        """
        seq_number, batch_size, input_channel, height, width = inputs.size()
        # Flatten sequence and batch for spatial convolutions
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        if self.use_checkpoint:
            inputs    = checkpoint(self.stems_wrapper,     inputs, self.dummy_tensor)
            cls_feat  = checkpoint(self.cls_convs_wrapper, inputs, self.dummy_tensor)
            cls_output = checkpoint(self.cls_preds_wrapper, cls_feat, self.dummy_tensor)
            reg_feat  = checkpoint(self.reg_convs_wrapper, inputs, self.dummy_tensor)
            reg_output = checkpoint(self.reg_preds_wrapper, reg_feat, self.dummy_tensor)
        else:
            inputs     = self.stems(inputs)
            cls_feat   = self.cls_convs(inputs)
            cls_output = self.cls_preds(cls_feat)   # (B*S, 1, H, W)
            reg_feat   = self.reg_convs(inputs)
            reg_output = self.reg_preds(reg_feat)   # (B*S, 1, H, W)

        # Apply sigmoid if not already applied by the final activation
        if self.acts[-2] == "":
            cls_output = cls_output.sigmoid()

        # Zero out depth predictions in cells classified as dry (< cls_thred)
        # This suppresses false positive floods in building cells / domain edges
        reg_output = self.correction_depth(reg_output, cls_output, self.cls_thred)

        # Concatenate: [depth, wet_prob] along channel dim, then restore (S, B, 2, H, W)
        outputs = torch.cat([reg_output, cls_output], 1)  # (B*S, 2, H, W)
        outputs = torch.reshape(
            outputs,
            (seq_number, batch_size, outputs.size(1), outputs.size(2), outputs.size(3)),
        )

        return outputs  # (S, B, 2, H, W)

    def correction_depth(self, reg_output_t, cls_output_t, flood_thres=0.5):
        """Apply classification mask to depth predictions.

        Cells where ``cls_output_t < flood_thres`` (predicted dry) have their
        depth set to zero.  This enforces consistency between the two heads:
        a dry-classified cell will always predict zero depth, regardless of
        the regression branch output.

        Parameters
        ----------
        reg_output_t : Tensor, shape (B, 1, H, W)
            Raw regression output (predicted depth).
        cls_output_t : Tensor, shape (B, 1, H, W)
            Classification probability (wet likelihood).
        flood_thres : float
            Probability threshold (default: ``self.cls_thred = 0.5``).

        Returns
        -------
        Tensor, shape (B, 1, H, W)
            Depth predictions with dry cells zeroed out.
        """
        correction = (cls_output_t >= flood_thres).float()  # binary mask
        return reg_output_t * correction
