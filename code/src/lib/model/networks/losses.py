"""
losses.py — Physics-informed loss functions for U-RNN.

U-RNN predicts two quantities simultaneously for urban flood nowcasting:
  1. Flood EXTENT  (wet/dry classification) — FocalBCELoss
  2. Flood DEPTH   (regression in mm)       — WMSELoss (Weighted MSE)

These are combined into ``FocalBCE_and_WMSE`` with epoch-dependent weighting
that implements a curriculum learning strategy (Cao et al., J. Hydrology 2025):
  • epochs < 500 : prioritise classification (loss = reg + 10·cls)
      Classification (wet/dry extent) is an easier task — learning it first
      provides a reliable mask for depth prediction in later epochs.
  • epochs ≥ 500 : shift focus to depth regression (loss = reg + 0.1·cls)

Both loss components address the severe class imbalance of urban floods:
  • Dry cells dominate (~80% of the domain); naive MSE / BCE would predict
    everything dry and still achieve low loss.
  • FocalBCE down-weights easy dry-cell predictions so training focuses on
    uncertain wet/dry boundary cells.
  • WMSE up-weights flooded cells by a factor of 20 to counter the sparsity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def select_loss_function(loss_name, reduction):
    """Return the loss function instance for the given name.

    Parameters
    ----------
    loss_name : str
        Currently only ``"FocalBCE_and_WMSE"`` is supported.
    reduction : str
        ``"mean"`` or ``"sum"``.
    """
    if loss_name == "FocalBCE_and_WMSE":
        return FocalBCE_and_WMSE(reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name!r}")


class FocalBCE_and_WMSE(nn.Module):
    """Combined loss: Focal BCE (flood extent) + Weighted MSE (flood depth).

    The two components target the two output heads of ``YOLOXHead``:
      • cls head → FocalBCELoss  (wet=1 / dry=0 per cell)
      • reg head → WMSELoss      (flood depth in mm, normalised to [0,1])

    Epoch-dependent weighting applies curriculum learning:
      epoch < 500 : L = L_reg + 10 * L_cls   (learn extent first)
      epoch ≥ 500 : L = L_reg + 0.1 * L_cls  (refine depth)

    Parameters
    ----------
    gamma : float
        Focal modulation exponent (default 2, same as RetinaNet).
    alpha : float
        Balance factor for the positive (wet) class (default 0.25).
    reg_weight : float
        Unused legacy parameter kept for API compatibility.
    reduction : str
        ``"mean"`` or ``"sum"``.
    """

    def __init__(self, gamma=2, alpha=0.25, reg_weight=2, reduction='mean'):
        super(FocalBCE_and_WMSE, self).__init__()
        self.reg_losses = WMSELoss(reduction=reduction)
        self.cls_loss = FocalBCELoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.reg_weight = reg_weight
        self.reduction = reduction

    def forward(self, inputs, targets, epoch):
        """Compute the combined loss.

        Parameters
        ----------
        inputs : dict
            ``{"cls": Tensor(B,T,H,W), "reg": Tensor(B,T,H,W)}``
            Both are normalised to [0, 1].
        targets : Tensor, shape (B, T, H, W)
            Ground-truth flood depth in mm, normalised to [0, 1].
        epoch : int
            Current training epoch — controls the cls/reg balance.

        Returns
        -------
        dict
            ``{"loss", "loss_reg", "loss_reg_label", "loss_reg_pred", "loss_cls"}``
        """
        # Convert depth targets to binary wet/dry labels for the cls head
        cls_targets = self.label_reg2cls(targets)  # 1 where depth > 0
        loss_cls = self.cls_loss(inputs["cls"], cls_targets)

        # Weighted MSE for depth regression
        loss_reg, loss_reg_flood, loss_reg_unflood = self.reg_losses(
            inputs["reg"], targets)

        # Curriculum weighting: learn extent before depth
        if epoch < 500:
            loss = loss_reg + 10 * loss_cls   # classification-heavy phase
        else:
            loss = loss_reg + 0.1 * loss_cls  # regression-heavy phase

        return {
            "loss": loss,
            "loss_reg": loss_reg,
            "loss_reg_label": loss_reg_flood,   # MSE on flooded cells
            "loss_reg_pred": loss_reg_unflood,  # MSE on dry cells
            "loss_cls": loss_cls,
        }

    def label_reg2cls(self, reg_targets):
        """Convert continuous depth targets to binary wet/dry labels.

        Any cell with depth > 0 (after normalisation, > 0 still) is wet (1).
        Dry cells (depth == 0) become 0.
        """
        return (reg_targets > 0).float()


class WMSELoss(nn.Module):
    """Weighted Mean Squared Error for flood depth regression.

    Urban flood grids are dominated by dry cells (depth = 0). A naive MSE
    would be driven almost entirely by the dry-cell terms, making it easy to
    minimise by predicting zero everywhere. This loss up-weights the flooded
    cells by a factor of 20 to force accurate depth prediction where it matters.

        L_reg = factor * MSE(pred[wet], target[wet])
              +         MSE(pred[dry], target[dry])
        (factor = 20, calibrated to the ~80:20 dry:wet cell ratio)

    Parameters
    ----------
    reduction : str
        ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(self, reduction='mean'):
        super(WMSELoss, self).__init__()
        self.factor = 20   # wet-cell up-weight to counter ~80% dry-cell dominance
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute weighted MSE.

        Parameters
        ----------
        inputs : Tensor, shape (B, T, H, W)
            Predicted normalised flood depth.
        targets : Tensor, shape (B, T, H, W)
            Ground-truth normalised flood depth.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            (total_loss, wet_cell_mse, dry_cell_mse)
        """
        flood_inputs, flood_targets, unflood_inputs, unflood_targets = \
            self.cal_mask(inputs, targets)

        # MSE separately for wet and dry cells
        flood_loss = F.mse_loss(flood_inputs, flood_targets,
                                reduction=self.reduction)
        unflood_loss = F.mse_loss(unflood_inputs, unflood_targets,
                                  reduction=self.reduction)

        loss = self.factor * flood_loss + unflood_loss

        return loss, flood_loss, unflood_loss

    def cal_mask(self, inputs, targets):
        """Split inputs and targets into flooded and dry subsets.

        The wet/dry split is based on the *target* depth so the weighting
        reflects the ground-truth spatial extent, not the model's prediction.
        """
        flood_mask = targets.gt(0)    # True where target depth > 0 (wet cells)
        unflood_mask = targets.le(0)  # True where target depth == 0 (dry cells)
        return (
            torch.masked_select(inputs, flood_mask),
            torch.masked_select(targets, flood_mask),
            torch.masked_select(inputs, unflood_mask),
            torch.masked_select(targets, unflood_mask),
        )


class FocalBCELoss(nn.Module):
    """Focal Binary Cross-Entropy Loss for wet/dry flood extent classification.

    Standard BCE is biased toward the majority class (dry cells ≈ 80% of domain).
    Focal loss (Lin et al., 2017) addresses this by multiplying each sample's loss
    by a modulation factor that down-weights well-classified easy examples:

        L_cls = - α · (1-p̂)^γ · y · log(p̂)
                - (1-α) · p̂^γ · (1-y) · log(1-p̂)

        p̂  : predicted wet probability (sigmoid output)
        y   : binary wet/dry label
        α   : 0.25 — balance factor favouring the minority (wet) class
        γ   : 2.0  — focusing parameter; γ=0 → standard BCE

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher γ → more focus on hard examples.
    alpha : float
        Weight for the positive (wet) class.
    reduction : str
        ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets, inf=1e-9):
        """Compute the Focal BCE loss.

        Parameters
        ----------
        inputs : Tensor
            Predicted wet-cell probabilities in [0, 1] (after sigmoid).
        targets : Tensor
            Binary wet/dry labels (0 or 1), same shape as inputs.
        inf : float
            Small epsilon for numerical stability inside log().

        Returns
        -------
        Tensor
            Scalar focal BCE loss.
        """
        pt = inputs

        # Focal BCE formula (Lin et al. 2017, Eq. 5)
        loss = (
            - self.alpha * (1 - pt) ** self.gamma * targets * torch.log(abs(pt) + inf)
            - (1 - self.alpha) * pt ** self.gamma * (1 - targets) * torch.log(abs(1 - pt) + inf)
        )

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss
