import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
# from FuseNet0902 import CrossWITBLOCK, Config
from CrossAttention import LightweightCrossAttention, LSTM, LSTM_batch, WindowAttention
from carafe import CARAFE_channel
# from YOLO import YOLOv7FeatureExtractor
# from END import ENDFeatureExtractor

class Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU()
        self.BN = nn.BatchNorm2d(ch_out)
    def forward(self, x):
        out = self.conv1(x)
        out = self.BN(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.prelu2(out)
        out = self.BN(out)

        return out

class Block1(nn.Module):
    def __init__(self, input, output):
        super(Block1, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0)
        self.ReLU = nn.ReLU()
        self.bath = nn.BatchNorm2d(output)

    def forward(self, x):
        out = self.conv(x)
        output0 = self.bath(out)
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

class FCSA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        # assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x

class UpSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),nn.ReLU(),
            # nn.UpsamplingNearest2d(scale_factor = 2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.f(x)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, input, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = input.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * input + beta
        return x

def norm(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

class FUSENET(nn.Module):
    def __init__(self,
                 dim=48, hidden_dim=768, clip_dim=768, num_tokens=77):
        super(FUSENET, self).__init__()

        # self.model_clip = model_clip
        # self.model_clip.eval()

        # CLIP encoders
        # self.image_encoder = model_clip.visual
        # self.text_encoder = model_clip.textual

        # 任务描述文本编码
        # self.task_text = "This is an infrared and visible fusion image task."
        # self.task_embed = self.text_encoder(self.task_text)  # [1, L, C]

        self.prompt_guidance = FeatureWiseAffine(in_channels=768, out_channels=128)
        self.cross_att = WindowAttention()
        # # 提示池
        # self.detail_prompt_pool = nn.Parameter(torch.randn(num_tokens, clip_dim))
        # self.target_prompt_pool = nn.Parameter(torch.randn(num_tokens, clip_dim))
        #
        # # 提示融合层
        # self.prompt_fusion = nn.Sequential(nn.Linear(3 * clip_dim, clip_dim), nn.LayerNorm(clip_dim), nn.ReLU())
        # # 特征融合控制器
        # self.fusion_controller = nn.Sequential(
        #     nn.Linear(clip_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.Sigmoid()
        # )

        self.reduce_channel = Block1(128, 64)
        self.up01 = CARAFE_channel(128, 64)
        self.up02 = CARAFE_channel(128, 64)

        self.up11 = CARAFE_channel(128, 64)
        self.up12 = CARAFE_channel(128, 64)

        self.LSTM2 = LSTM_batch(input_channels=64, hidden_channels=64, kernel_size=1, height=16, width=16)
        self.LSTM1 = LSTM_batch(64, 64, 1, 8, 8)
        self.LSTM3 = LSTM_batch(128, 128, 1, 16, 16)

        # self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
        # self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
        # self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)

        self.decoder = nn.Sequential(
            # nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 2, 3, 1, 1), nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d()

    def forward(self, fA04, fA14, fA24, fB04, fB14, fB24, imgA_V, imgB_V):
        ###########################################################################################################################1
        self.dropout = nn.Dropout2d(0.1)

        # prompt = self.prompt_generate(imgA_V, imgB_V)

        tpA01 = self.up01(fA24, fA14)  # + fA14
        tpB01 = self.up11(fB24, fB14)  # + fB14
        cross_att10 = self.LSTM1(tpA01, tpB01, 8)
        cross_att20 = self.LSTM1(tpB01, tpA01, 8)
        tpA02 = fA14 + tpA01 + cross_att10
        tpB02 = fB14 + tpB01 + cross_att20

        tpA03 = self.up02(tpA02, fA04)  # + fA04
        tpB03 = self.up12(tpB02, fB04)  # + fB04
        cross_att11 = self.LSTM2(tpA03, tpB03, 16)
        cross_att21 = self.LSTM2(tpB03, tpA03, 16)
        tpA04 = fA04 + tpA03 + cross_att11
        tpB04 = fB04 + tpB03 + cross_att21

        f_cat = torch.cat((tpA04, tpB04), dim=1)
        # f_prompt = self.prompt_guidance(f_cat, prompt) + f_cat
        # f = torch.cat((f_cat, f_prompt), dim=1)  #直接Cat
        # f_att,_= self.cross_att(f_cat,f_prompt)  #窗口注意力
        # f_att = self.LSTM3(f_cat,f_prompt, 16)         #使用LSTM实现交叉
        F = self.decoder(f_cat)
        Wir, Wvi = torch.split(F, [1, 1], dim=1)
        Fusion = Wir * imgA_V + Wvi * imgB_V

        return Fusion

    # @torch.no_grad()
    # def get_text_feature(self, text):
    #     text_feature = self.model_clip.encode_text(text)
    #     return text_feature