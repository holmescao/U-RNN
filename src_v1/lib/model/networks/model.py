from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from src.lib.model.networks.decoder import Decoder
from src.lib.model.networks.encoder import Encoder
from src.lib.model.networks.head.flood_head import YOLOXHead

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class activation:
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == "leaky":
            return F.leaky_relu(input,
                                negative_slope=self.negative_slope,
                                inplace=self.inplace)
        elif self._act_type == "relu":
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == "sigmoid":
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class ModuleWrapperIgnoresCnn(nn.Module):
    def __init__(self, module):
        super().__init__()
        # self.module = module.to("cuda:2")
        self.module = module

    def forward(self, x, dummy_arg=None):
        # 这里向前传播的时候, 不仅传入x, 还传入一个有梯度的变量, 但是没有参与计算
        assert dummy_arg is not None
        x = self.module(x)
        return x
    
class Drainage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv_drainage = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

        self.dummy_tensor = torch.ones(1,
                                       dtype=torch.float32,
                                       requires_grad=True)
        
        self.wrapper_conv_drainage = ModuleWrapperIgnoresCnn(self.conv_drainage)
        
    def forward(self, x):
        
        x = checkpoint(self.wrapper_conv_drainage,
                                  x, self.dummy_tensor)
        
        return x
    
class Reg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.reg_conv = nn.Conv2d(in_channels=16,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        
        self.use_checkpoint = True

        self.reg_conv_module_wrapper = ModuleWrapperIgnoresCnn(
            self.reg_conv)
        

        self.dummy_tensor = torch.ones(1,
                                       dtype=torch.float32,
                                       requires_grad=True)
        
    def forward(self, x):
        
        seq_number, batch_size, input_channel, height, width = x.size()

        # 把img展开成seq * bs 张图片
        x = torch.reshape(x, (-1, input_channel, height, width))

        x = checkpoint(self.reg_conv_module_wrapper,
                                  x, self.dummy_tensor)
        
        x = torch.reshape(
            x,
            (seq_number, batch_size, x.size(1), x.size(2),
             x.size(3)),
        )

        return x
        


class ED(nn.Module):
    """
    decoder最后一层后再输出最终结果
    """

    def __init__(self, clstm_flag,
                 encoder_params,
                 decoder_params,
                 cls_thred=0.5,
                 use_checkpoint=True):
        super().__init__()

        self.encoder = Encoder(clstm_flag,
                               encoder_params[0],
                               encoder_params[1],
                               use_checkpoint=use_checkpoint)
        self.decoder = Decoder(clstm_flag,
                               decoder_params[0],
                               decoder_params[1],
                               use_checkpoint=use_checkpoint)
        self.head = YOLOXHead(cls_thred)
        # self.head = Reg()

        # self.risk_conv = nn.Conv2d(in_channels=3,
        #                            out_channels=1,
        #                            kernel_size=filter_size,
        #                            stride=1,
        #                            padding=(filter_size - 1) // 2)

        # self.use_checkpoint = use_checkpoint
        # self.dummy_tensor = torch.ones(1,
        #                                dtype=torch.float32,
        #                                requires_grad=True)
        # self.risk_conv_module_wrapper = ModuleWrapperIgnores2ndArg(
        #     self.risk_conv)

        

        

    def forward(self, input_t, prev_encoder_state, prev_decoder_state):
        input_t = input_t.permute(1, 0, 2, 3, 4)  # to S,B,C,64,64
        # check_uniformity(input_t)
        
        # # S*B,C,64,64
        # seq_number, batch_size, input_channel, height, width = input_t.size()
        # input_t = torch.reshape(input_t, (-1, input_channel, height, width))
        # # 先处理成风险变量
        # risk_t = checkpoint(self.risk_conv_module_wrapper, input_t[:, :3],
        #                     self.dummy_tensor)
        # input_t = torch.cat([risk_t, input_t[:, 3:4]], dim=1)  # 风险变量+降雨
        # # S*B,2,64,64
        # # 把img的shape转回来
        # input_t = torch.reshape(
        #     input_t,
        #     (seq_number, batch_size, input_t.size(1), input_t.size(2),
        #      input_t.size(3)),
        # )

        encoder_state_t = self.encoder(input_t, prev_encoder_state)
        # check_uniformity(encoder_state_t[0]) 
        # check_uniformity(encoder_state_t[1]) 
        # check_uniformity(encoder_state_t[2]) 
        
        # (B, S, F, H, W)
        output_t, decoder_state_t = self.decoder(
            encoder_state_t, prev_decoder_state)  # (B, S, F, H, W)
        # check_uniformity(output_t)
        # check_uniformity(decoder_state_t[0])
        # check_uniformity(decoder_state_t[1])
        # check_uniformity(decoder_state_t[2])
        
        
        output_t = self.head(output_t)
        
        # reg_output_t = self.head(output_t)
        # check_uniformity(reg_output_t)
        # reg_output_t = self.reg_conv_module_wrapper(output_t)

        reg_output_t, cls_output_t = output_t[:, :, 0:1], output_t[:, :, 1:]
        
        return reg_output_t, cls_output_t, encoder_state_t, decoder_state_t

    # def correction_depth(self, reg_output_t, cls_output_t, flood_thres=0.5):
    #     # TODO: 改成attention的方式，即可训练
    #     # 法1：把cls转成0-1变量，然后加权到reg
    #     correction = (cls_output_t >= flood_thres).float()  # TODO: 可能要去掉
    #     # correction = (cls_output_t.sigmoid()>=flood_thres).float()
    #     # print(reg_output_t.max())
    #     reg_output_t = reg_output_t * correction
    #     # print(reg_output_t.max())
    #     # TODO: 法2：把cls直接加权到reg

    #     return reg_output_t