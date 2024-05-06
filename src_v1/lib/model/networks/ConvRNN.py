#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ConvRNN.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   convrnn cell
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
# from main import DEVICE

# device = DEVICE
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
        assert dummy_arg is not None
        x = self.module(x)
        return x


class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """

    def __init__(self, use_checkpoint, shape, input_channels, filter_size,
                 num_features, module):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = int(input_channels)
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = int(num_features)
        self.padding = (filter_size - 1) // 2
        
        self.module = module
        # self.module = "encoder" # TODO: 临时测试

        if self.module == "encoder":  # 只考虑encoder的state
            conv_input_channels = self.input_channels + self.num_features
            conv_output_channels = 2 * self.num_features
        elif self.module == "decoder":  # 同时考虑encoder和decoder的state
            conv_input_channels = self.input_channels + 2*self.num_features
            conv_output_channels = 2 * self.num_features
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                conv_input_channels,
                conv_output_channels,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(conv_output_channels // 32, conv_output_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                conv_input_channels,
                self.num_features,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(self.num_features // 32, self.num_features),
        )

        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1,
                                       dtype=torch.float32,
                                       requires_grad=True)
        self.conv1_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv1)
        self.conv2_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv2)

    def forward(self, inputs=None, hidden_state=None, seq_len=1):
        # TODO: S,B,C,H,W
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []

        # 开始遍历
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]
            
            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            if self.use_checkpoint:
                gates = checkpoint(self.conv1_module_wrapper, combined_1,
                                   self.dummy_tensor)
            else:
                gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            # zgate = gates[:, :self.num_features]
            # rgate = gates[:, self.num_features:]
            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            # h' = tanh(W*(x+r*H_t-1))
            if self.module == "encoder":
                combined_2 = torch.cat((x, r * htprev), 1)
            elif self.module == "decoder":
                etprev, dtprev = torch.split(htprev, self.num_features, dim=1)
                combined_2 = torch.cat((x, etprev, r * dtprev), 1)

            if self.use_checkpoint:
                ht = checkpoint(self.conv2_module_wrapper, combined_2,
                                self.dummy_tensor)
            else:
                ht = self.conv2(combined_2)  # num_features
            ht = torch.tanh(ht)

            if self.module == "encoder":
                htnext = (1 - z) * htprev + z * ht
            elif self.module == "decoder":
                # ! decoder的state用于遗忘
                # decoder_htprev = htprev[:, htprev.shape[1]//2:]
                htnext = (1 - z) * dtprev + z * ht
                
            output_inner.append(htnext)
            htprev = htnext  # 这句话在这里其实无效，因为seq_len=1

        return torch.stack(output_inner)
        # return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell"""

    def __init__(self, use_checkpoint, shape, input_channels, filter_size,
                 num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.input_channels + self.num_features,
                4 * self.num_features,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(  # 分组归一化，每组32个channels
                4 * self.num_features // 32, 4 * self.num_features),
        )

        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")

        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1,
                                       dtype=torch.float32,
                                       requires_grad=True)
        self.module_wrapper = ModuleWrapperIgnores2ndArg(self.conv)

    # def forward(self, inputs_state=(None, None), seq_len=1):
    def forward(self, inputs=None, hidden_state=None, seq_len=1):
        """Encoding/Decoding Sequence

        将input sequence输入到CNNLSTM结构中，
        得到hidden_state的输出和最终的cell

        """

        # inputs, hidden_state = inputs_state
        # 初始化hidden state和cell
        # if hidden_state is None:
        if hidden_state[0] is None:
            hx = torch.zeros(  # (B, num_features, H, W)
                inputs.size(1), self.num_features, self.shape[0],
                self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        # 迭代预测
        index = 0
        # for index in range(seq_len):
        if inputs is None:  # decoding的时候x一直是0
            x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                            self.shape[1]).cuda()
        else:
            # 卷积的结果
            x = inputs[index, ...]  # (B, cnn_out_c, H, W)

        # 将输入x和隐层状态h_{t-1}一起做卷积
        combined = torch.cat((x, hx), 1)

        if self.use_checkpoint:
            gates = checkpoint(self.module_wrapper, combined,
                               self.dummy_tensor)
        else:
            gates = self.conv(combined)  # gates: S, num_features*4, H, W

        # 每个gate都有自己的weight，所以直接预测了4份，也就是用的时候要划分4份
        # it should return 4 tensors: i,f,c,o
        ingate, forgetgate, cellgate, outgate = torch.split(gates,
                                                            self.num_features,
                                                            dim=1)
        """ 
        更新LSTM的核心
        
        其中，
        cell和hidden变量中: x表示t-1时刻; y表示t时刻
        """
        # TODO：ingate,forgetgate,outgate都没有与C_{t-1}结合
        # LSTM的核心逻辑
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cy = (forgetgate * cx) + ingate * torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        hy = outgate * torch.tanh(cy)

        # 添加hidden state
        output_inner.append(hy)

        # update hidden state：h_{t-1}, h_{t}; c_{t-1}, c_{t}
        hx = hy
        cx = cy

        # 一次预测多个时序时才需要！
        # return [torch.stack(output_inner), (hy, cy)]

        output_inner.append(cy)
        return torch.stack(output_inner)
