#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
# from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

# from yolox.utils import bboxes_iou, meshgrid

# from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

from torch.utils.checkpoint import checkpoint, checkpoint_sequential

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.2, inplace=inplace)
    elif name == "gelu":
        module = nn.GELU()
    elif name == "sigmoid":
        module = nn.Sigmoid()
    elif name == "":
        module = None
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def get_normlization(name, num_features):
    if name == "bn":
        module = nn.BatchNorm2d(num_features)
    elif name == "gn":
        group = 1 if num_features//32 == 0 else num_features//32
        module = nn.GroupNorm(group, num_features)
    elif name == "":
        module = None
    else:
        raise AttributeError("Unsupported normlization type: {}".format(name))
    return module


class finalConv(nn.Module):
    """A Conv2d -> silu/leaky relu block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride,
        groups=1,
        bias=False,
        act="leaky",
        norm="gn",
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            # groups=groups,
            # bias=bias,
        )

        # self.bn = nn.BatchNorm2d(out_channels)
        self.norm = get_normlization(norm, out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        if self.act is not None:
            if self.norm is not None:
                return self.act(self.norm(self.conv(x)))
            else:
                return self.act(self.conv(x))
        else:
            if self.norm is not None:
                return self.norm(self.conv(x))
            else:
                return self.conv(x)
        # if self.act is not None:
        #     if self.bn is not None:
        #         return self.act(self.bn(self.conv(x)))
        #     else:
        #         return self.act(self.conv(x))
        # else:
        #     if self.bn is not None:
        #         return self.bn(self.conv(x))
        #     else:
        #         return self.conv(x)

    # def fuseforward(self, x):
    #     return self.act(self.conv(x))


class ModuleWrapperIgnores2ndArg_cnn(nn.Module):
    def __init__(self, module):
        super().__init__()
        # self.module = module.to("cuda:2")
        self.module = module

    def forward(self, x, dummy_arg=None):
        # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
        assert dummy_arg is not None
        x = self.module(x)
        return x


class YOLOXHead(nn.Module):
    def __init__(
        self,
        cls_thred=0.5,
        in_channels=64,
        width=0.25,
        depthwise=False,
        use_checkpoint=True,
    ):
        """
        1层stem卷积, in=16, out=256, k=1
        2层cls和reg卷积, in=256, out=256, k=1
        1层cls预测卷积, in=256, out=1, k=1
        1层reg预测卷积, in=256, out=1, k=1


        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        Conv = DWConv if depthwise else BaseConv

        # acts = ["silu"] * 3 + [""]*2
        # acts = ["lrelu"] * 3 + [""]*2
        # acts = ["silu"] * 3 + ["silu"]*2
        # acts = ["silu"] * 3 + ["sigmoid", "silu"]
        acts = ["silu"] * 3 + ["sigmoid", "lrelu"]
        # acts = ["lrelu"] * 3 + ["sigmoid", "lrelu"]
        # acts = ["silu"] * 3 + ["sigmoid", "relu"]
        self.acts = acts

        # 跟在backbone后的
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
            # bias=True,
            act=self.acts[3],
            norm="",
        )

        # final reg pred
        self.reg_preds = finalConv(
            in_channels=int(in_channels * width),
            out_channels=1,
            ksize=1,
            stride=1,
            # bias=True,
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

        self.cls_thred=cls_thred

    def forward(self, inputs):
        seq_number, batch_size, input_channel, height, width = inputs.size()

        # 把img展开成seq * bs 张图片
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        if self.use_checkpoint:
            inputs = checkpoint(self.stems_wrapper, inputs, self.dummy_tensor)
            # class branch
            cls_feat = checkpoint(self.cls_convs_wrapper,
                                  inputs, self.dummy_tensor)
            cls_output = checkpoint(self.cls_preds_wrapper,
                                    cls_feat, self.dummy_tensor)

            # cls_output = checkpoint(self.cls_preds_wrapper,
            #                         inputs, self.dummy_tensor)

            # reg branch
            reg_feat = checkpoint(self.reg_convs_wrapper, inputs,
                                  self.dummy_tensor)
            reg_output = checkpoint(self.reg_preds_wrapper, reg_feat,
                                    self.dummy_tensor)

            # reg_output = checkpoint(self.reg_preds_wrapper, inputs,
            #                         self.dummy_tensor)
        else:
            # inputs = self.stems(inputs)

            # class branch
            # cls_feat = self.cls_convs(inputs)
            # cls_output = self.cls_preds(cls_feat)  # B, 1, h, w

            # # reg branch
            # reg_feat = self.reg_convs(inputs)
            # reg_output = self.reg_preds(reg_feat)  # B, 1, h, w
            # reg branch
            reg_output = self.reg_preds(inputs)  # B, 1, h, w

        if self.acts[-2] == "":  # 需要手动进行simgoid
            cls_output = cls_output.sigmoid()

        # reg correlation
        reg_output = self.correction_depth(reg_output, cls_output, self.cls_thred)

        outputs = torch.cat([reg_output, cls_output], 1)  # B, 2, h, w
        # outputs = reg_output

        # 把img的shape转回来
        outputs = torch.reshape(
            outputs,
            (seq_number, batch_size, outputs.size(1), outputs.size(2),
             outputs.size(3)),
        )

        return outputs

    def correction_depth(self, reg_output_t, cls_output_t, flood_thres=0.5):
        # TODO: 改成attention的方式，即可训练
        # 法1：把cls转成0-1变量，然后加权到reg
        correction = (cls_output_t >= flood_thres).float()  # TODO: 可能要去掉
        # correction = (cls_output_t.sigmoid()>=flood_thres).float()
        # print(reg_output_t.max())
        reg_output_t = reg_output_t * correction
        # print(reg_output_t.max())
        # TODO: 法2：把cls直接加权到reg

        return reg_output_t
