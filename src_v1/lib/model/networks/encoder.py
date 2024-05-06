#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   encoder.py
@Time    :   2020/03/09 18:47:50
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   encoder
"""

from torch import nn
from src.lib.model.networks.utils import make_layers
import torch
import logging
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class ModuleWrapperIgnores2ndArg_lstm(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    # def forward(self, x):
    #     # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
    #     # assert dummy_arg is not None
    #     x = self.module(x[0])
    #     return x
    def forward(self, x, hx, cx, dummy_arg=None):
        # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
        assert dummy_arg is not None
        x = self.module(x, (hx, cx))
        return x


class ModuleWrapperIgnores2ndArg_gru(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    # def forward(self, x):
    #     # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
    #     # assert dummy_arg is not None
    #     x = self.module(x[0])
    #     return x
    def forward(self, x, hx, dummy_arg=None):
        # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
        assert dummy_arg is not None
        x = self.module(x, hx)
        return x


class Encoder(nn.Module):
    def __init__(self, clstm, subnets, rnns, use_checkpoint):
        super().__init__()
        """__init__ 根据设计的net结构，构建net

        每一层都是：卷积层 + ConvRNN cell
        """

        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        self.stage1 = make_layers(subnets[0])
        self.stage2 = make_layers(subnets[1])
        self.stage3 = make_layers(subnets[2])
        self.rnn1 = rnns[0]
        self.rnn2 = rnns[1]
        self.rnn3 = rnns[2]

        self.use_checkpoint = use_checkpoint
        self.clstm = clstm  # flag

        self.dummy_tensor = torch.ones(1,
                                       dtype=torch.float32,
                                       requires_grad=True)
        # self.dummy_tensor = torch.ones(
        #     1, dtype=torch.float32, requires_grad=True).to(device)
        self.stage1_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage1)
        self.stage2_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage2)
        self.stage3_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage3)
        if self.clstm:
            self.rnn1_wrapper = ModuleWrapperIgnores2ndArg_lstm(self.rnn1)
            self.rnn2_wrapper = ModuleWrapperIgnores2ndArg_lstm(self.rnn2)
            self.rnn3_wrapper = ModuleWrapperIgnores2ndArg_lstm(self.rnn3)
        else:
            self.rnn1_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn1)
            self.rnn2_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn2)
            self.rnn3_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn3)

    def forward_by_stage(self, i, inputs, hidden_state, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        """Conv层的操作"""
        # 把img展开成seq * bs 张图片
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        # 输入网络
        if self.use_checkpoint:
            if i == 1:
                inputs = checkpoint(self.stage1_wrapper, inputs,
                                    self.dummy_tensor)
            elif i == 2:
                inputs = checkpoint(self.stage2_wrapper, inputs,
                                    self.dummy_tensor)
            elif i == 3:
                inputs = checkpoint(self.stage3_wrapper, inputs,
                                    self.dummy_tensor)

        else:
            inputs = subnet(inputs)
        """RNN层的操作"""
        # 把img的shape转回来
        inputs = torch.reshape(
            inputs,
            (seq_number, batch_size, inputs.size(1), inputs.size(2),
             inputs.size(3)),
        )
        # 把inputs放进ConvRNN网络
        if self.clstm:
            if self.use_checkpoint:
                hidden_state = (None,
                                None) if hidden_state == None else hidden_state
                if i == 1:
                    outputs_state_stage = checkpoint(self.rnn1_wrapper, inputs,
                                                     hidden_state[0],
                                                     hidden_state[1],
                                                     self.dummy_tensor)
                elif i == 2:
                    outputs_state_stage = checkpoint(self.rnn2_wrapper, inputs,
                                                     hidden_state[0],
                                                     hidden_state[1],
                                                     self.dummy_tensor)
                elif i == 3:
                    outputs_state_stage = checkpoint(self.rnn3_wrapper, inputs,
                                                     hidden_state[0],
                                                     hidden_state[1],
                                                     self.dummy_tensor)
            else:
                outputs_state_stage = rnn(inputs, hidden_state)
        else:
            if self.use_checkpoint:
                # hidden_state = (
                #     None, None) if hidden_state == None else hidden_state
                if i == 1:
                    outputs_state_stage = checkpoint(self.rnn1_wrapper, inputs,
                                                     hidden_state,
                                                     self.dummy_tensor)
                elif i == 2:
                    outputs_state_stage = checkpoint(self.rnn2_wrapper, inputs,
                                                     hidden_state,
                                                     self.dummy_tensor)
                elif i == 3:
                    outputs_state_stage = checkpoint(self.rnn3_wrapper, inputs,
                                                     hidden_state,
                                                     self.dummy_tensor)
            else:
                outputs_state_stage = rnn(inputs, hidden_state)

        if self.clstm:  # LSTM
            hy, cy = outputs_state_stage[0], outputs_state_stage[1]
            outputs_stage = hy.unsqueeze(0)
            state_stage = (hy, cy)
        else:  # GRU
            hy = outputs_state_stage[0]
            outputs_stage = hy.unsqueeze(0)
            state_stage = (hy)

        return outputs_stage, state_stage

    def forward(self, inputs, state_stages):
        """获取所有变量"""

        hidden_states = []
        # 遍历每一个stage
        for i in range(1, self.blocks + 1):
            # stage_{i-1}的hidden state作为stage_{i}的input
            inputs, state_stage = self.forward_by_stage(
                i, inputs, state_stages[i - 1],
                getattr(self, "stage" + str(i)), getattr(self, "rnn" + str(i)))

            # 添加每一stage最终的hidden state，用于接下来的decoding
            hidden_states.append(state_stage)

        return tuple(hidden_states)
        # return torch.as_tensor(tuple(hidden_states))  # TODO:改为torch
