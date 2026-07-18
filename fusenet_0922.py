import torch
import torch.nn as nn
from CrossAttention import GSC_batch_3

class Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.prelu2(out)
        return out
class Block1(nn.Module):
    def __init__(self, input, output):
        super(Block1, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0)
        self.ReLU = nn.ReLU()
        self.bath = nn.BatchNorm2d(output)

    def forward(self, x):
        out = self.conv(x)
        output0 = self.bath(out)  ###去掉
        output = self.ReLU(output0)
        return output
class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, scales=[1, 2]):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0) for _ in scales
        ])

    def forward(self, x):
        attention_maps = []
        for scale, attention_layer in zip(self.scales, self.attention_layers):
            attention_map = attention_layer(x)
            attention_map = torch.sigmoid(torch.mean(attention_map, dim=1, keepdim=True))
            # attention_map = torch.nn.functional.interpolate(attention_map, scale_factor=scale, mode='nearest')
            attention_maps.append(attention_map)

        out = sum(attention_maps) * x
        return out
class UpSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.UpsamplingNearest2d(scale_factor = 2),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.f(x)

class FUSENET(nn.Module):
    def __init__(self):
        super(FUSENET, self).__init__()

        self.L00 = Block(32, 64)
        self.L01 = Block(64, 32)
        self.L04 = Block(64, 32)
        self.L06 = Block(32, 16)
        self.L08 = Block(16, 1)

        self.L10 = Block(256, 128)
        self.L11 = Block(128, 64)
        self.L12 = Block(64, 32)
        self.L13 = Block(32, 16)

        self.L20 = Block(256, 128)
        self.L21 = Block(128, 64)
        self.L22 = Block(64, 32)
        self.L23 = Block(32, 16)

        self.u04 = UpSamplingBlock(128, 128)
        self.u05 = UpSamplingBlock(64, 64)
        self.u06 = UpSamplingBlock(32, 32)
        self.u07 = UpSamplingBlock(16, 16)

        self.u14 = UpSamplingBlock(128, 128)
        self.u15 = UpSamplingBlock(64, 64)
        self.u16 = UpSamplingBlock(32, 32)
        self.u17 = UpSamplingBlock(16, 16)

        self.MSA00 = MultiScaleAttention(128)
        self.MSA01 = MultiScaleAttention(64)
        self.MSA02 = MultiScaleAttention(32)
        self.MSA03 = MultiScaleAttention(16)
        self.MSA10 = MultiScaleAttention(128)
        self.MSA11 = MultiScaleAttention(64)
        self.MSA12 = MultiScaleAttention(32)
        self.MSA13 = MultiScaleAttention(16)

        self.LSTM1 = GSC_batch_3(32, 32, 1, 16, 16)
        self.LSTM2 = GSC_batch_3(32, 32, 1, 16, 16)
        self.LSTM3 = GSC_batch_3(16, 16, 1, 16, 16)
        self.LSTM4 = GSC_batch_3(16, 16, 1, 16, 16)


        self.down_conv01 = Block(64, 32)
        self.down_conv11 = Block(64, 32)
        self.down_conv02 = Block(128, 64)
        self.down_conv12 = Block(128, 64)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d()

    def forward(self, fA04, fA14, fA24, fB04, fB14, fB24, f_P_ir_16, f_P_ir_32, f_P_ir_64, f_P_vis_16, f_P_vis_32, f_P_vis_64, imgA_V, imgB_V):
        ###########################################################################################################################1
        self.dropout = nn.Dropout2d(0.1)

        tpA06 = self.L04(fA04)

        tpA12 = self.L12(fA14)
        tpA22 = self.L22(fA24)
        tpB06 = self.L04(fB04)

        tpB12 = self.L12(fB14)
        tpB22 = self.L22(fB24)

        sA27 = tpA22
        sA17 = tpA12 + self.MSA12(self.u16(sA27))
        sA07 = tpA06 + self.MSA02(self.u06(sA17))

        sB27 = tpB22
        sB17 = tpB12 + self.MSA12(self.u16(sB27))
        sB07 = tpB06 + self.MSA02(self.u06(sB17))
########################提示注入##############################################
        EX05 = self.LSTM1(sA07, f_P_ir_32, sB07, 16)
        tpA07 = sA07 + EX05
        EX06 = self.LSTM2(sB07, f_P_vis_32, sA07, 16)
        tpB07 = sB07 + EX06

        tpA09 = self.L06(tpA07)
        tpA13 = self.L13(sA17)
        tpA23 = self.L23(sA27)
        tpB09 = self.L06(tpB07)
        tpB13 = self.L13(sB17)
        tpB23 = self.L23(sB27)

        sA28 = tpA23
        sA18 = tpA13 + self.MSA13(self.u17(sA28))
        sA08 = tpA09 + self.MSA03(self.u07(sA18))
        sB28 = tpB23
        sB18 = tpB13 + self.MSA13(self.u17(sB28))
        sB08 = tpB09 + self.MSA03(self.u07(sB18))
#########################注入提示#############################################
        EX07 = self.LSTM3(sA08, f_P_ir_16, sB08, 16)
        tpA010 = sA08 + EX07
        EX08 = self.LSTM4(sB08, f_P_vis_16, sA08, 16)
        tpB010 = sB08 + EX08

####################################################################
        tpA012 = self.L08(tpA010)
        tpB012 = self.L08(tpB010)

        weight_ir = self.sigmoid(tpA012)
        weight_vis = self.sigmoid(tpB012)

        Fusion = imgA_V * weight_ir + weight_vis * imgB_V

        return Fusion
