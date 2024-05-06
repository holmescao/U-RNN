import torch
import torch.nn as nn
import torch.nn.functional as F


def select_loss_function(loss_name, reduction):
    """
    Selects and returns a loss function based on the given name.

    Parameters:
    - loss_name: Name of the loss function to use.
    - reduction: Specifies the reduction to apply to the output, 'mean' or 'sum'.

    Returns:
    - nn.Module: An instance of the requested loss function.
    """
    if loss_name == "FocalBCE_and_WMSE":
        return FocalBCE_and_WMSE(reduction=reduction)
    else:
        raise ValueError("Unsupported loss function")


class FocalBCE_and_WMSE(nn.Module):
    """
    Combines Focal Binary Cross-Entropy and Weighted Mean Squared Error for loss calculation.
    """

    def __init__(self, gamma=2, alpha=0.25, reg_weight=2, reduction='mean'):
        """
        Initializes the FocalBCE_and_WMSE loss module with parameters for both focal loss and weighted MSE.

        Parameters:
        - gamma: Focusing parameter for Focal BCE to adjust the rate at which easy examples are down-weighted.
        - alpha: Balancing factor for Focal BCE to balance the importance of positive/negative examples.
        - reg_weight: Weight factor for the regression loss component.
        - reduction: Specifies the method for reducing the loss over the batch; can be 'none', 'mean', or 'sum'.
        """
        super(FocalBCE_and_WMSE, self).__init__()

        self.reg_losses = WMSELoss(reduction=reduction)
        self.cls_loss = FocalBCELoss(gamma=gamma,
                                     alpha=alpha,
                                     reduction=reduction)

        self.reg_weight = reg_weight
        self.reduction = reduction

    def forward(self, inputs, targets, epoch):
        """
        Calculates the combined loss for classification and regression tasks.

        Parameters:
        - inputs: Contains 'cls' for classification inputs and 'reg' for regression inputs.
        - targets : The ground truth values.
        - epoch: Current epoch number to adjust loss components dynamically.

        Returns:
        - dict: Contains detailed loss components including combined loss.
        """
        cls_targets = self.label_reg2cls(targets)
        loss_cls = self.cls_loss(inputs["cls"], cls_targets)

        loss_reg, loss_reg_flood, loss_reg_unflood = self.reg_losses(
            inputs["reg"], targets)

        if epoch < 500:
            loss = loss_reg + 10 * loss_cls
        else:
            loss = loss_reg + 0.1 * loss_cls

        return {
            "loss": loss,
            "loss_reg": loss_reg,
            "loss_reg_label": loss_reg_flood,
            "loss_reg_pred": loss_reg_unflood,
            "loss_cls": loss_cls,
        }

    def label_reg2cls(self, reg_targets):
        """
        Converts regression targets to binary classification targets.

        Parameters:
        - reg_targets: Regression targets.

        Returns:
        - Tensor: Binary classification targets.
        """
        return (reg_targets > 0).float()


class WMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss that gives different weights to certain parts of the data.
    """

    def __init__(self, reduction='mean'):
        """
        Initializes the Weighted Mean Squared Error Loss module.

        Parameters:
        - reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
        """
        super(WMSELoss, self).__init__()

        self.factor = 20
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calculates weighted mean squared error between inputs and targets.

        Parameters:
        - inputs: Predicted values.
        - targets: Ground truth values.

        Returns:
        - tuple: Total loss, flood-specific loss, and unflood-specific loss.
        """
        flood_inputs, flood_targets, unflood_inputs, unflood_targets \
            = self.cal_mask(inputs, targets)

        flood_loss = F.mse_loss(flood_inputs,
                                flood_targets,
                                reduction=self.reduction)
        unflood_loss = F.mse_loss(unflood_inputs,
                                  unflood_targets,
                                  reduction=self.reduction)

        loss = self.factor * flood_loss + unflood_loss

        return loss, flood_loss, unflood_loss

    def cal_mask(self, inputs, targets):
        """
        Separates inputs and targets into flood and unflood regions.

        Parameters:
        - inputs: Inputs tensor.
        - targets: Targets tensor.

        Returns:
        - tuple: Tensors for flood inputs, flood targets, unflood inputs, and unflood targets.
        """
        flood_mask = targets.gt(0)
        unflood_mask = targets.le(0)
        return (
            torch.masked_select(inputs, flood_mask),
            torch.masked_select(targets, flood_mask),
            torch.masked_select(inputs, unflood_mask),
            torch.masked_select(targets, unflood_mask)
        )


class FocalBCELoss(nn.Module):
    """
    Focal Binary Cross-Entropy Loss to focus training on hard examples and down-weight easy negatives.
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        """
        Initializes the Focal Binary Cross-Entropy Loss module.

        Parameters:
        - gamma: Modulating factor to adjust the rate at which easy examples are down-weighted.
        - alpha: Weighting factor for the positive class in the binary classification.
        - reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
        """
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets, inf=1e-9):
        """
        Calculates the focal BCE loss for binary classification tasks.

        Parameters:
        - inputs: Predicted probabilities.
        - targets: Ground truth binary labels.

        Returns:
        - Tensor: The computed focal BCE loss.
        """
        pt = inputs

        loss = - self.alpha * (1 - pt) ** self.gamma * targets * torch.log(abs(pt)+inf) - \
            (1 - self.alpha) * pt ** self.gamma * \
            (1 - targets) * torch.log(abs(1 - pt)+inf)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss
