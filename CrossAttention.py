import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_

class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x1, x2):

        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x1.shape
        query = x1.flatten(2).transpose(1, 2)  # (B, H*W, C)
        key = x2.flatten(2).transpose(1, 2)  # (B, H*W, C)
        value = x2.flatten(2).transpose(1, 2)  # (B, H*W, C)

        q = self.q_linear(query).reshape(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(key).reshape(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(value).reshape(B, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x = self.proj(x).transpose(1, 2).reshape(B, C, H, W)
        x = x.permute(0, 2, 3,1).contiguous()
        x = self.norm(x)
        x = self.proj_drop(x)

        return x

class LightweightCrossAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16, pool_size=None):
        super().__init__()
        self.n_head = n_head
        self.pool_size = pool_size
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

        if pool_size is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        if self.pool_size is not None:
            input1 = x_A
            input2 = x_B
            x_A = self.avgpool(x_A)
            x_B = self.avgpool(x_B)
            _, _, pooled_height, pooled_width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, -1)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, -1)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum("bnci, bncj -> bnij", query_B, key_A).contiguous() / math.sqrt(channel)
        attn_A = torch.softmax(attn_A, -1)

        out_A = torch.einsum("bnij, bncj -> bnci", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, pooled_height, pooled_width))
        out_A = F.interpolate(out_A, size=(height, width), mode='bilinear', align_corners=False)
        out_A = out_A + input1

        attn_B = torch.einsum(
            "bnci, bncj -> bnij", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = torch.softmax(attn_B, -1)

        out_B = torch.einsum("bnij, bncj -> bnci", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, pooled_height, pooled_width))
        out_B = F.interpolate(out_B, size=(height, width), mode='bilinear', align_corners=False)
        out_B = out_B + input2

        return out_A, out_B

class Cross_attention_xuhan(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

class WindowAttention(nn.Module):
    def __init__(self, dim = 128, window_size = 8, num_heads = 8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.network_depth = 16

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.V = nn.Conv2d(dim, dim, 1)
        self.ass_V = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_ass = nn.Conv2d(dim, dim, 1)
        self.QK = nn.Conv2d(dim, 2 * dim, 1)
        self.ass_QK = nn.Conv2d(dim, 2 * dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # def check_size(self, x, shift=False):
    #     _, _, h, w = x.size()
    #     mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
    #     mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
    #
    #     if shift:
    #         x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
    #                       self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
    #     else:
    #         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    #     return x

    def forward(self, image_feature, prompt_feature):
        B, C, H, W = image_feature.shape
        V = self.V(image_feature)
        ass_V = self.ass_V(prompt_feature);

        QK = self.QK(image_feature)
        ass_QK = self.ass_QK(prompt_feature)

        QKV = torch.cat([QK, V], dim=1)
        ass_QKV = torch.cat([ass_QK, ass_V], dim=1)

        # shift
        # shifted_QKV = self.check_size(QKV, self.shift_size > 0)
        # shifted_ass_QKV = self.check_size(ass_QKV, self.shift_size > 0)
        # Ht, Wt = shifted_QKV.shape[2:]

        # partition windows
        shifted_QKV = QKV.permute(0, 2, 3, 1)
        shifted_ass_QKV = ass_QKV.permute(0, 2, 3, 1)

        qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
        ass_qkv = window_partition(shifted_ass_QKV, self.window_size)  # nW*B, window_size**2, C

        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        ass_qkv = ass_qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        ass_q, ass_k, ass_v = ass_qkv[0], ass_qkv[1], ass_qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # text modality -> vision
        ass_q = ass_q * self.scale
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))

        # vision -> text modality

        ass_attn = (ass_q @ ass_k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)
        ass_attn = ass_attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        ass_attn = self.softmax(ass_attn)

        attn_windows = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        ass_attn_windows = (ass_attn @ ass_v).transpose(1, 2).reshape(B_, N, self.dim)

        # merge windows
        out = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        ass_out = window_reverse(ass_attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        # out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
        # ass_out = ass_shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]

        attn_out = out.permute(0, 3, 1, 2)
        ass_attn_out = ass_out.permute(0, 3, 1, 2)

        return attn_out, ass_attn_out

class GSC_batch_2(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, height=16, width=16):
        super(GSC_batch_2, self).__init__()
        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = input_channels
        self.kernel_size = kernel_size

        self.height = height
        self.width = width

        self.padding = int((kernel_size - 1) / 2)

        # 第一个时间步的权重矩阵
        self.Wxi1 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)  # Concat后的输入
        self.Wxf1 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Wxc1 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Wxo1 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        
        # 第二个时间步的权重矩阵
        self.Wxi2 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)  # Concat后的输入
        self.Whi2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf2 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc2 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo2 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # Cell state的权重参数
        self.Wci1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wcf1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wco1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wci2 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wcf2 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wco2 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))

    def forward(self, outputA, outputB, block_size):
        # -----------------------------裁剪图片-----------------------------------------------
        # 保存原始张量的形状信息
        B, C, H, W = outputA.size()
        block_size = block_size
        num_blocks_h = H // block_size
        num_blocks_w = W // block_size

        A_reshape_tensor = outputA.view(B, C, num_blocks_h, block_size, num_blocks_w, block_size)
        A_reshape_tensor = A_reshape_tensor.permute(0, 2, 4, 1, 3, 5).contiguous()
        outputA = A_reshape_tensor.view(B * num_blocks_h * num_blocks_w, C, block_size, block_size)

        B_reshape_tensor = outputB.view(B, C, num_blocks_h, block_size, num_blocks_w, block_size)
        B_reshape_tensor = B_reshape_tensor.permute(0, 2, 4, 1, 3, 5).contiguous()
        outputB = B_reshape_tensor.view(B * num_blocks_h * num_blocks_w, C, block_size, block_size)
        # -----------------------------裁剪图片-----------------------------------------------

        # 第一个时间步：Concat inputA
        concat_input1 = torch.cat([outputA, outputB], dim=1)  # 模拟Concat操作
        
        # 第一个时间步的门控计算
        i1 = torch.sigmoid(self.Wxi1(concat_input1))
        f1 = torch.sigmoid(self.Wxf1(concat_input1))
        c1_tilde = torch.tanh(self.Wxc1(concat_input1))
        o1 = torch.sigmoid(self.Wxo1(concat_input1))
        
        # 第一个时间步的cell state和hidden state
        # 实现图片中的"1-"操作：(1-f1) * initial_c + i1 * c1_tilde
        # 当initial_c为0时，简化为：i1 * c1_tilde
        c1 = (1 - f1) * 0 + i1 * c1_tilde  # 明确实现"1-"操作
        h1 = o1 * torch.tanh(c1)
        
        # 第二个时间步：Concat inputB和h_{t-1}
        concat_input2 = torch.cat([outputB, h1], dim=1)  # Concat操作
        
        # 第二个时间步的门控计算
        i2 = torch.sigmoid(self.Wxi2(concat_input2) + self.Whi2(h1) + c1 * self.Wci2)
        f2 = torch.sigmoid(self.Wxf2(concat_input2) + self.Whf2(h1) + c1 * self.Wcf2)
        c2_tilde = torch.tanh(self.Wxc2(concat_input2) + self.Whc2(h1))
        o2 = torch.sigmoid(self.Wxo2(concat_input2) + self.Who2(h1) + c1 * self.Wco2)
        
        # 第二个时间步的cell state和hidden state
        c2 = f2 * c1 + i2 * c2_tilde
        h2 = o2 * torch.tanh(c2)

        # --------------------------复原图片----------------------------------------------------
        dh = h2.view(B, num_blocks_h, num_blocks_w, C, block_size, block_size)
        dh = dh.permute(0, 3, 1, 4, 2, 5).contiguous()
        dh = dh.view(B, C, H, W)
        # --------------------------复原图片----------------------------------------------------

        return dh

class GSC_batch_3(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, height=16, width=16):
        super(GSC_batch_3, self).__init__()
        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = input_channels
        self.kernel_size = kernel_size

        self.height = height
        self.width = width

        self.padding = int((kernel_size - 1) / 2)

        # 第一个时间步的权重矩阵
        self.Wxi1 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding,
                              bias=True)  # Concat后的输入
        self.Wxf1 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding,
                              bias=True)
        self.Wxc1 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding,
                              bias=True)
        self.Wxo1 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding,
                              bias=True)

        # 第二个时间步的权重矩阵
        self.Wxi2 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding,
                              bias=True)  # Concat后的输入
        self.Whi2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf2 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding,
                              bias=True)
        self.Whf2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc2 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding,
                              bias=True)
        self.Whc2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo2 = nn.Conv2d(self.input_channels * 2, self.hidden_channels, self.kernel_size, 1, self.padding,
                              bias=True)
        self.Who2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # Cell state的权重参数
        self.Wci1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wcf1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wco1 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wci2 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wcf2 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wco2 = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))

    def forward(self, outputA, outputB, outputC, block_size):
        # -----------------------------裁剪图片-----------------------------------------------
        # 保存原始张量的形状信息
        B, C, H, W = outputA.size()
        block_size = block_size
        num_blocks_h = H // block_size
        num_blocks_w = W // block_size

        A_reshape_tensor = outputA.view(B, C, num_blocks_h, block_size, num_blocks_w, block_size)
        A_reshape_tensor = A_reshape_tensor.permute(0, 2, 4, 1, 3, 5).contiguous()
        outputA = A_reshape_tensor.view(B * num_blocks_h * num_blocks_w, C, block_size, block_size)

        B_reshape_tensor = outputB.view(B, C, num_blocks_h, block_size, num_blocks_w, block_size)
        B_reshape_tensor = B_reshape_tensor.permute(0, 2, 4, 1, 3, 5).contiguous()
        outputB = B_reshape_tensor.view(B * num_blocks_h * num_blocks_w, C, block_size, block_size)

        C_reshape_tensor = outputC.view(B, C, num_blocks_h, block_size, num_blocks_w, block_size)
        C_reshape_tensor = C_reshape_tensor.permute(0, 2, 4, 1, 3, 5).contiguous()
        outputC = C_reshape_tensor.view(B * num_blocks_h * num_blocks_w, C, block_size, block_size)
        # -----------------------------裁剪图片-----------------------------------------------

        # 第一个时间步：Concat inputA
        concat_input1 = torch.cat([outputA, outputB], dim=1)  # 模拟Concat操作

        # 第一个时间步的门控计算
        i1 = torch.sigmoid(self.Wxi1(concat_input1))
        f1 = torch.sigmoid(self.Wxf1(concat_input1))
        c1_tilde = torch.tanh(self.Wxc1(concat_input1))
        o1 = torch.sigmoid(self.Wxo1(concat_input1))

        # 第一个时间步的cell state和hidden state
        # 实现图片中的"1-"操作：(1-f1) * initial_c + i1 * c1_tilde
        # 当initial_c为0时，简化为：i1 * c1_tilde
        c1 = (1 - f1) * 0 + i1 * c1_tilde  # 明确实现"1-"操作
        h1 = o1 * torch.tanh(c1)

        # 第二个时间步：Concat inputB和h_{t-1}
        concat_input2 = torch.cat([outputC, h1], dim=1)  # Concat操作

        # 第二个时间步的门控计算
        i2 = torch.sigmoid(self.Wxi2(concat_input2) + self.Whi2(h1) + c1 * self.Wci2)
        f2 = torch.sigmoid(self.Wxf2(concat_input2) + self.Whf2(h1) + c1 * self.Wcf2)
        c2_tilde = torch.tanh(self.Wxc2(concat_input2) + self.Whc2(h1))
        o2 = torch.sigmoid(self.Wxo2(concat_input2) + self.Who2(h1) + c1 * self.Wco2)

        # 第二个时间步的cell state和hidden state
        c2 = f2 * c1 + i2 * c2_tilde
        h2 = o2 * torch.tanh(c2)

        # --------------------------复原图片----------------------------------------------------
        dh = h2.view(B, num_blocks_h, num_blocks_w, C, block_size, block_size)
        dh = dh.permute(0, 3, 1, 4, 2, 5).contiguous()
        dh = dh.view(B, C, H, W)
        # --------------------------复原图片----------------------------------------------------

        return dh
