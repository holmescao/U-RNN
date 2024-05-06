#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   decoder.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   decoder
"""

from torch import nn
from src.lib.model.networks.utils import make_layers
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class ModuleWrapperIgnores2ndArg_cnn(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    # def forward(self, x):
    #     # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
    #     # assert dummy_arg is not None
    #     x = self.module(x[0])
    #     return x
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
    def forward(self, hx, cx, x,  dummy_arg=None):
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
    def forward(self, hx, x,  dummy_arg=None):
        # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
        assert dummy_arg is not None
        x = self.module(x, hx)
        return x


class Decoder(nn.Module):
    def __init__(self, clstm, subnets, rnns, use_checkpoint):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        self.stage3 = make_layers(subnets[0])
        self.stage2 = make_layers(subnets[1])
        self.stage1 = make_layers(subnets[2])
        self.rnn3 = rnns[0]
        self.rnn2 = rnns[1]
        self.rnn1 = rnns[2]

        self.clstm = clstm  # flag
        self.use_checkpoint = use_checkpoint

        self.dummy_tensor = torch.ones(
            1, dtype=torch.float32, requires_grad=True)
        self.stage1_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage1)
        self.stage2_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage2)
        self.stage3_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage3)

        self.rnn1_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn1)
        self.rnn2_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn2)
        self.rnn3_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn3)

        
        
    def forward_by_stage(self, i, inputs, encoder_states, decoder_states=None):
        # TODO：检查到这里
        """forward_by_stage 解码1个stage

        先经过CLSTM，再用卷积层（deconv）

        Args:
            inputs (_type_): _description_
            state (_type_): _description_
            subnet (_type_): _description_
            rnn (_type_): _description_

        Returns:
            _type_: _description_
        """
        # ConvRNN
        if decoder_states is None:
            decoder_states = torch.zeros_like(encoder_states).cuda()
        # state = encoder_states
        state = torch.cat((encoder_states, decoder_states),
                          dim=1)
        
        if self.use_checkpoint:
            if i == 1:
                outputs_state_stage = checkpoint(self.rnn1_wrapper,
                                                state, inputs,  self.dummy_tensor)
            elif i == 2:
                outputs_state_stage = checkpoint(self.rnn2_wrapper,
                                                state, inputs,  self.dummy_tensor)
            elif i == 3:
                outputs_state_stage = checkpoint(self.rnn3_wrapper,
                                                state, inputs,  self.dummy_tensor)
        else:
            if i == 1:
                outputs_state_stage = self.rnn1(inputs,state)
            if i == 2:
                outputs_state_stage = self.rnn2(inputs,state)
            if i == 3:
                outputs_state_stage = self.rnn3(inputs,state)

        hy = outputs_state_stage[0]
        inputs = hy.unsqueeze(0)
        state_stage = (hy)
        
        # 卷积层
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        if self.use_checkpoint:
            if i == 1:
                inputs = checkpoint(self.stage1_wrapper,
                                    inputs, self.dummy_tensor)
            elif i == 2:
                inputs = checkpoint(self.stage2_wrapper,
                                    inputs, self.dummy_tensor)
            elif i == 3:
                inputs = checkpoint(self.stage3_wrapper,
                                    inputs, self.dummy_tensor)
        else:
            if i == 1:
                inputs = self.stage1(inputs)
            if i == 2:
                inputs = self.stage2(inputs)
            if i == 3:
                inputs = self.stage3(inputs)


        inputs = torch.reshape(
            inputs,
            (seq_number, batch_size, inputs.size(
                1), inputs.size(2), inputs.size(3)),
        )
        
        
        return inputs, state_stage

        # input: 5D S*B*C*H*W

    def forward(self, encoder_states, decoder_states):
        hidden_states = []
        # 3 -> 2 -> 1
        # 先解码最后一层
        
        inputs, state = self.forward_by_stage(
            3, None,
            encoder_states[-1],
            decoder_states[0],
        )
        # check_uniformity(inputs)
        # check_uniformity(state)
        
        hidden_states.append(state)
        for i in list(range(1, self.blocks))[::-1]:
            inputs, state = self.forward_by_stage(
                i, inputs,
                encoder_states[i - 1],
                decoder_states[self.blocks-i],
            )

            hidden_states.append(state)

            
        inputs = inputs.transpose(0, 1)  # to B,S,num_feats,h,w
        
        
        return inputs, tuple(hidden_states)


def check_uniformity(tensor):
    """
    Check if all elements in the last two dimensions of the tensor are the same.

    Args:
    tensor (torch.Tensor): A PyTorch tensor.

    Returns:
    bool: True if all elements in the last two dimensions are the same, False otherwise.
    """
    # Flatten the last two dimensions and compare each element with the first element
    last_two_dims_flattened = tensor.view(-1, tensor.size(-2) * tensor.size(-1))
    first_element = last_two_dims_flattened[:, 0].unsqueeze(1)
    if torch.all(last_two_dims_flattened == first_element, dim=1).all():
        print("相等")
    else:
        print("不相等")