import torch
import torch.nn as nn
from .network_blocks import BaseConv, DWConv, finalConv
from torch.utils.checkpoint import checkpoint


class ModuleWrapperIgnores2ndArg_cnn(nn.Module):
    """
    A module wrapper that ignores an additional dummy argument during the forward pass.
    This is typically used to maintain compatibility with certain frameworks or for gradient checkpointing.
    """

    def __init__(self, module):
        """
        Initializes the ModuleWrapperIgnores2ndArg_cnn with the specified module.

        Parameters:
        - module: The module to wrap.
        """
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        """
        Forwards the input through the module while ignoring the dummy argument.

        Parameters:
        - x: The input tensor.
        - dummy_arg: A dummy argument that must be provided but is not used.

        Returns:
        - Tensor: The output from the module.
        """
        assert dummy_arg is not None, "dummy_arg is required but was None"
        return self.module(x)


class YOLOXHead(nn.Module):
    """
    YOLOXHead processes inputs through separate branches for classification (cls) and 
    regression (reg) tasks, applying convolutions and using checkpoints if required.
    """

    def __init__(
        self,
        cls_thred=0.5,
        in_channels=64,
        width=0.25,
        depthwise=False,
        use_checkpoint=True,
    ):
        """
        Initializes the YOLOXHead with specific configurations for processing features.

        Parameters:
        - cls_thred: Threshold value for classification output.
        - in_channels: Number of input channels.
        - width: Scaling factor for the number of channels; adjusts channel capacity.
        - depthwise: Whether to use depthwise convolution.
        - use_checkpoint: Whether to use gradient checkpointing to save memory during training.
        """
        super().__init__()

        Conv = DWConv if depthwise else BaseConv

        # Define activation functions to be used in layers
        acts = ["silu"] * 3 + ["sigmoid", "lrelu"]

        self.acts = acts

        # Stem layer
        self.stems = BaseConv(
            in_channels=int(in_channels * width),
            out_channels=int(in_channels * width),
            ksize=1,
            stride=1,
            act=acts[0],
        )

        # class branch
        self.cls_convs = nn.Sequential(*[
            Conv(
                in_channels=int(in_channels * width),
                out_channels=int(in_channels * width),
                ksize=1,
                stride=1,
                act=acts[1],
            ),
            Conv(
                in_channels=int(in_channels * width),
                out_channels=int(in_channels * width),
                ksize=1,
                stride=1,
                act=acts[1],
            ),
        ])

        # reg branch
        self.reg_convs = nn.Sequential(*[
            Conv(
                in_channels=int(in_channels * width),
                out_channels=int(in_channels * width),
                ksize=1,
                stride=1,
                act=acts[2],
            ),
            Conv(
                in_channels=int(in_channels * width),
                out_channels=int(in_channels * width),
                ksize=1,
                stride=1,
                act=acts[2],
            ),
        ])

        # final class pred
        self.cls_preds = finalConv(
            in_channels=int(in_channels * width),
            out_channels=1,
            ksize=1,
            stride=1,
            act=self.acts[3],
            norm="",
        )

        # final reg pred
        self.reg_preds = finalConv(
            in_channels=int(in_channels * width),
            out_channels=1,
            ksize=1,
            stride=1,
            act=self.acts[4],
            norm="",
        )

        self.stems_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stems)
        self.cls_convs_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.cls_convs)
        self.reg_convs_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.reg_convs)
        self.cls_preds_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.cls_preds)
        self.reg_preds_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.reg_preds)

        self.dummy_tensor = torch.ones(1,
                                       dtype=torch.float32,
                                       requires_grad=True)

        self.use_checkpoint = use_checkpoint

        self.cls_thred = cls_thred

    def forward(self, inputs):
        """
        Forward pass through the YOLOX head, applying convolutional and activation layers,
        with optional checkpointing.

        Parameters:
        - inputs: Input tensor to be processed through the head.

        Returns:
        - Tensor: The processed output tensor containing both classification and regression results.
        """
        # Reshape for processing: flatten the sequence and batch dimensions
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        # Apply layers with optional checkpointing
        if self.use_checkpoint:
            inputs = checkpoint(self.stems_wrapper, inputs, self.dummy_tensor)
            # class branch
            cls_feat = checkpoint(self.cls_convs_wrapper,
                                  inputs, self.dummy_tensor)
            cls_output = checkpoint(self.cls_preds_wrapper,
                                    cls_feat, self.dummy_tensor)
            # reg branch
            reg_feat = checkpoint(self.reg_convs_wrapper, inputs,
                                  self.dummy_tensor)
            reg_output = checkpoint(self.reg_preds_wrapper, reg_feat,
                                    self.dummy_tensor)
        else:
            inputs = self.stems(inputs)
            # class branch
            cls_feat = self.cls_convs(inputs)
            cls_output = self.cls_preds(cls_feat)  # B, 1, h, w
            # reg branch
            reg_feat = self.reg_convs(inputs)
            reg_output = self.reg_preds(reg_feat)  # B, 1, h, w

        # Apply sigmoid if necessary
        if self.acts[-2] == "":
            cls_output = cls_output.sigmoid()

        # Apply regression correction based on classification threshold
        reg_output = self.correction_depth(
            reg_output, cls_output, self.cls_thred)

        # Concatenate and reshape outputs to original dimensions
        outputs = torch.cat([reg_output, cls_output], 1)  # B, 2, h, w
        outputs = torch.reshape(
            outputs,
            (seq_number, batch_size, outputs.size(1), outputs.size(2),
             outputs.size(3)),
        )

        return outputs

    def correction_depth(self, reg_output_t, cls_output_t, flood_thres=0.5):
        """
        Applies a threshold-based correction to regression outputs based on classification outputs.

        Parameters:
        - reg_output: Regression output tensor.
        - cls_output: Classification output tensor.
        - flood_thres: Classification threshold to apply.

        Returns:
        - Tensor: Corrected regression output.
        """
        correction = (cls_output_t >= flood_thres).float()

        reg_output_t = reg_output_t * correction

        return reg_output_t
