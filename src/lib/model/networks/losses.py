from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def select_loss_function(loss_name, reduction):
    if loss_name == "MSE":
        loss_function = nn.MSELoss(reduction=reduction)
    elif loss_name == "Focal_MSE":
        loss_function = FocalMSELoss(reduction=reduction)
    elif loss_name == "WMSE":
        loss_function = WMSELoss(reduction=reduction)
    elif loss_name == "Focal_MSE_BCE":
        loss_function = FocalMSEBCELoss(reduction=reduction)
    elif loss_name == "FocalBCE_and_Flood_MSE":
        loss_function = FocalBCE_and_Flood_MSE(reduction=reduction)
    elif loss_name == "FocalBCE_and_WMSE":
        loss_function = FocalBCE_and_WMSE(reduction=reduction)
    elif loss_name == "FocalBCE_and_W2MSE":
        loss_function = FocalBCE_and_W2MSE(reduction=reduction)

    return loss_function


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size).cuda()

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).cuda()
        return gauss/gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.create_window(self.window_size, channel)
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class W2MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(W2MSELoss, self).__init__()

        self.factor = 0.8
        self.reduction = reduction

    def forward(self, inputs, targets):
        flood_inputs, flood_targets, unflood_inputs, unflood_targets \
            = self.cal_mask(inputs, targets)

        flood_loss = F.mse_loss(flood_inputs,
                                flood_targets,
                                reduction=self.reduction)
        unflood_loss = F.mse_loss(unflood_inputs,
                                  unflood_targets,
                                  reduction=self.reduction)

        loss = flood_loss + self.factor * unflood_loss

        return loss, flood_loss, unflood_loss

    def cal_mask(self, inputs, targets, thred=10/2190, penalty=5):
        # pred中积水小于0的部分：增大权重
        inputs = torch.where(inputs < 0, inputs.abs() * penalty, inputs)

        # label中积水和无积水的区别对待
        flood_mask = targets.gt(thred)  # > thred
        flood_inputs = torch.masked_select(inputs, flood_mask)
        flood_targets = torch.masked_select(targets, flood_mask)

        unflood_mask = targets.le(thred)  # <= thred
        unflood_inputs = torch.masked_select(inputs, unflood_mask)
        unflood_targets = torch.masked_select(targets, unflood_mask)

        return flood_inputs, flood_targets, unflood_inputs, unflood_targets


class WMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WMSELoss, self).__init__()

        self.factor = 20
        self.reduction = reduction

        # self.mae_loss = nn.L1Loss()

    def forward(self, inputs, targets):
        flood_inputs, flood_targets, unflood_inputs, unflood_targets \
            = self.cal_mask(inputs, targets)

        flood_loss = F.mse_loss(flood_inputs,
                                flood_targets,
                                reduction=self.reduction)
        unflood_loss = F.mse_loss(unflood_inputs,
                                  unflood_targets,
                                  reduction=self.reduction)

        # loss = flood_loss +  0.8 *unflood_loss
        loss = self.factor * flood_loss +  unflood_loss

        return loss, flood_loss, unflood_loss

    def cal_mask(self, inputs, targets):
        flood_mask = targets.gt(0)  # 有积水的部分
        flood_inputs = torch.masked_select(inputs, flood_mask)
        flood_targets = torch.masked_select(targets, flood_mask)

        unflood_mask = targets.le(0)  # 无积水的部分
        unflood_inputs = torch.masked_select(inputs, unflood_mask)
        unflood_targets = torch.masked_select(targets, unflood_mask)

        return flood_inputs, flood_targets, unflood_inputs, unflood_targets


class FloodMSELoss(nn.Module):
    """
    分别计算target和input中有flood的cell的loss的总和
    实际上等于2个部分，
    1. target中有flood的cell的loss
    2. model分类后认为有flood的cell的loss
    """

    def __init__(self, reduction='mean'):
        super(FloodMSELoss, self).__init__()

        self.reduction = reduction

    def forward(self, inputs, targets):
        flood_inputs_label, flood_targets_label, flood_inputs_pred, flood_targets_pred = \
            self.cal_mask(inputs, targets)

        loss_label = F.mse_loss(flood_inputs_label,
                                flood_targets_label,
                                reduction=self.reduction)
        loss_pred = F.mse_loss(flood_inputs_pred,
                               flood_targets_pred,
                               reduction=self.reduction)

        loss = loss_label + loss_pred

        return (
            loss,
            loss_label,
            loss_pred,
        )

    def cal_mask(self, inputs, targets):
        flood_mask_label = targets.gt(0)  # target认为有积水的部分
        flood_inputs_label = torch.masked_select(inputs, flood_mask_label)
        flood_targets_label = torch.masked_select(targets, flood_mask_label)

        flood_mask_pred = inputs.gt(0)  # pred认为有积水的部分
        flood_inputs_pred = torch.masked_select(inputs, flood_mask_pred)
        flood_targets_pred = torch.masked_select(targets, flood_mask_pred)

        return flood_inputs_label, flood_targets_label, flood_inputs_pred, flood_targets_pred


class FloodMSELoss_v1(nn.Module):
    """
    只计算target中有flood的cell的loss
    """

    def __init__(self, reduction='mean'):
        super(FloodMSELoss_v1, self).__init__()

        self.reduction = reduction

    def forward(self, inputs, targets):
        flood_inputs, flood_targets = self.cal_mask(inputs, targets)

        loss = F.mse_loss(flood_inputs,
                          flood_targets,
                          reduction=self.reduction)

        return loss

    def cal_mask(self, inputs, targets):
        flood_mask = targets.gt(0)  # 有积水的部分
        flood_inputs = torch.masked_select(inputs, flood_mask)
        flood_targets = torch.masked_select(targets, flood_mask)

        return flood_inputs, flood_targets


class FocalMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FocalMSELoss, self).__init__()

        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs, targets = self.cal_mask(inputs, targets)

        loss = F.mse_loss(inputs, targets, reduction=self.reduction)

        return loss

    def cal_mask(self, inputs, targets):
        flood_mask = targets.gt(0)  # 有积水的部分
        flood_inputs = torch.masked_select(inputs, flood_mask)
        flood_targets = torch.masked_select(targets, flood_mask)

        return flood_inputs, flood_targets


class FocalBCE_and_W2MSE(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reg_weight=2, reduction='mean'):
        super(FocalBCE_and_W2MSE, self).__init__()

        self.reg_losses = W2MSELoss(reduction=reduction)
        self.cls_loss = FocalBCELoss(gamma=gamma,
                                     alpha=alpha,
                                     reduction=reduction)

        self.reg_weight = reg_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # cls_targets = self.label_reg2cls(targets)
        # loss_cls = self.cls_loss(inputs["cls"], cls_targets)
        loss_reg, loss_reg_flood, loss_reg_unflood = self.reg_losses(
            inputs["reg"], targets)

        loss_cls = torch.zeros((1)).to(device)  # TODO: 实验用
        loss = self.reg_weight * loss_reg + loss_cls

        return {
            "loss": loss,
            "weight_loss_reg": self.reg_weight * loss_reg,
            "weight_loss_reg_label": self.reg_weight * loss_reg_flood,
            "weight_loss_reg_pred": self.reg_weight * loss_reg_unflood,
            "loss_reg": loss_reg,
            "loss_reg_label": loss_reg_flood,
            "loss_reg_pred": loss_reg_unflood,
            "loss_cls": loss_cls,
        }

    def label_reg2cls(self, reg_targets):
        """
        只分为有洪水1，无洪水0
        depth>0的为有洪水，否则为无洪水
        """
        return (reg_targets > 0).float()


class FocalBCE_and_WMSE(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reg_weight=2, reduction='mean'):
        super(FocalBCE_and_WMSE, self).__init__()

        self.reg_losses = WMSELoss(reduction=reduction)
        self.cls_loss = FocalBCELoss(gamma=gamma,
                                     alpha=alpha,
                                     reduction=reduction)
        
        self.ssim_loss = SSIMLoss()

        self.reg_weight = reg_weight
        self.reduction = reduction

        self.ssim_weight = 0.001
        
    def forward(self, inputs, targets, epoch):
        cls_targets = self.label_reg2cls(targets)
        loss_cls = self.cls_loss(inputs["cls"], cls_targets) # 预测的，是置信度
        
        # loss_ssim = self.ssim_loss(inputs["cls"][:,:,0], cls_targets[:,:,0])
        
        loss_reg, loss_reg_flood, loss_reg_unflood = self.reg_losses(
            inputs["reg"], targets)
        
        # # <400epoch cls:reg=1:1; >=400epoch cls:reg=1:200
        if epoch < 500:
            loss = loss_reg + 10 * loss_cls
        else:
            loss = loss_reg + 0.1 * loss_cls
        
        
        # loss = loss_reg + 10 * loss_cls
        # loss = self.reg_weight * loss_reg + loss_cls
        # loss = loss_reg + self.reg_weight * loss_cls + self.ssim_weight * loss_ssim

        return {
            "loss": loss,
            # "weight_loss_reg": self.reg_weight * loss_reg,
            # "weight_loss_reg_label": self.reg_weight * loss_reg_flood,
            # "weight_loss_reg_pred": self.reg_weight * loss_reg_unflood,
            "loss_reg": loss_reg,
            "loss_reg_label": loss_reg_flood,
            "loss_reg_pred": loss_reg_unflood,
            "loss_cls": loss_cls,
            "loss_ssim": torch.zeros(1).cuda(),
            # "loss_cls": torch.zeros(1).cuda(),
            # "loss_ssim": loss_ssim,
        }

    def label_reg2cls(self, reg_targets):
        """
        只分为有洪水1，无洪水0
        depth>0的为有洪水，否则为无洪水
        """
        return (reg_targets > 0).float()


class FocalBCE_and_Flood_MSE(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reg_weight=2, reduction='mean'):
        super(FocalBCE_and_Flood_MSE, self).__init__()

        self.reg_losses = WMSELoss(reduction=reduction)
        self.cls_loss = FocalBCELoss(gamma=gamma,
                                     alpha=alpha,
                                     reduction=reduction)

        self.reg_weight = reg_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # cls_targets = self.label_reg2cls(targets)
        # loss_cls = self.cls_loss(inputs["cls"], cls_targets)
        loss_reg, loss_reg_label, loss_reg_pred = self.reg_losses(
            inputs["reg"], targets)

        loss_cls = torch.zeros((1)).to(device)  # TODO: 实验用
        loss = self.reg_weight * loss_reg + loss_cls

        return {
            "loss": loss,
            "weight_loss_reg": self.reg_weight * loss_reg,
            "weight_loss_reg_label": self.reg_weight * loss_reg_label,
            "weight_loss_reg_pred": self.reg_weight * loss_reg_pred,
            "loss_reg": loss_reg,
            "loss_reg_label": loss_reg_label,
            "loss_reg_pred": loss_reg_pred,
            "loss_cls": loss_cls,
        }

    def label_reg2cls(self, reg_targets):
        """
        只分为有洪水1，无洪水0
        depth>0的为有洪水，否则为无洪水
        """
        return (reg_targets > 0).float()


class FocalMSEBCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, beta=1, reduction='mean'):
        super(FocalMSEBCELoss, self).__init__()

        self.FocalMSE = FocalMSELoss(reduction=reduction)
        self.FocalBCE = FocalBCELoss(gamma=gamma,
                                     alpha=alpha,
                                     reduction=reduction)

        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        # TODO：需要重新设计
        focal_bce = self.FocalBCE(self.norm(inputs), self.norm(targets))
        focal_mse = self.FocalMSE(inputs, targets)

        masked_input, _ = self.FocalMSE.cal_mask(inputs, targets)
        N = masked_input.numel()
        A = inputs.numel()
        sigma = 1 - N / A

        loss = (1 - sigma) * focal_bce + (self.beta + sigma) * focal_mse

        return loss

    def norm(self, data, thred=0.01):
        data = torch.where(data > thred, torch.ones_like(data),
                           torch.zeros_like(data))
        # mu = torch.mean(data)
        # std = torch.std(data)

        # data = (data - mu)/std
        return data


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs,
                                                          targets,
                                                          reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        """
        gamma是幂指数，越大表示对难分样本的权重越大
        alpha是平衡因子，取值在[0,1]之间
        """
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets, start_weight=10, inf=1e-9):
        # pt = torch.sigmoid(inputs)
        pt = inputs

        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * targets * torch.log(abs(pt)+inf) - \
            (1 - alpha) * pt ** self.gamma * \
            (1 - targets) * torch.log(abs(1 - pt)+inf)


        # # 为前三个时序赋予start_weight
        # weights = torch.ones_like(loss)
        # weights[:, :3, ...] *= start_weight
        # loss = loss * weights

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss
