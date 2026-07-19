import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import einops
from denet_feature_extractor import DENetFeatureExtractor
from LoadSegFeature1 import LoadSegFeature
from CrossAttention import GSC_batch_2

use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())

device = torch.device("cuda" if use_cuda else "cpu")


class Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()

        )
        self.BN = nn.BatchNorm2d(ch_out)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        fx = self.f(x)
        x = self.conv(x)
        x = self.BN(x)
        x = self.ReLU(x)
        output = fx + x
        # output = self.BN(output)
        # output = self.ReLU(output)
        return output

class Block1(nn.Module):
    def __init__(self, input, output):
        super(Block1, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0)
        self.ReLU = nn.ReLU()
        self.BN = nn.BatchNorm2d(output)

    def forward(self, x):
        out = self.conv(x)
        output = self.BN(out)
        output = self.ReLU(output)
        return output

class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.f(x)

class UpSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # nn.UpsamplingNearest2d(scale_factor = 2),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.f(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class Attention(nn.Module):
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
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=16):
        super(CAB, self).__init__()
        if is_light_sr:  # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, emb_size=64, patch_norm=True, norm_layer=nn.LayerNorm,
                 channel_first=False):
        super().__init__()
        self.in_chans = in_channels
        self.embed_dim = emb_size
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.norm_layer = norm_layer
        self.channel_first = channel_first
        self.conv = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.permute = Permute(0, 2, 3, 1)
        self.norm = norm_layer(emb_size) if patch_norm else nn.Identity()

    def forward(self, x):
        # Forward branch: upper-left -> lower-right.
        x = self.conv(x)
        x_forward = self.permute(x)  # [B, H_p, W_p, C]

        # Reverse branch: lower-right -> upper-left.
        # Flipping both spatial axes is equivalent to a 180-degree
        # reversal of the native 2D patch lattice.
        x_reverse = torch.flip(x_forward, dims=(1, 2)).contiguous()

        # The two sweeps are processed by the same WITRAN encoder,
        # so they share all recurrent parameters.
        x = torch.cat((x_forward, x_reverse), dim=0)
        x = self.norm(x)
        return x
class Unpatchify(nn.Module):
    def __init__(self, in_channels=4, patch_size=4, emb_size=64, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.in_chans = in_channels
        self.embed_dim = emb_size
        self.patch_size = patch_size
        self.permute = Permute(0, 3, 1, 2)
        self.conv_transpose = nn.ConvTranspose2d(emb_size, in_channels, kernel_size=patch_size, stride=patch_size,
                                                 bias=True)

    def forward(self, x):
        if not self.channel_first:
            x = self.permute(x)  # [2B, C, H_p, W_p]

        if x.shape[0] % 2 != 0:
            raise ValueError(
                "The batch dimension must be even because it contains "
                "paired forward and reverse water-wave sweeps."
            )

        x_forward, x_reverse = torch.chunk(x, chunks=2, dim=0)

        # Align the reverse-sweep representation with the original
        # spatial coordinates before aggregating the two directions.
        x_reverse = torch.flip(x_reverse, dims=(2, 3)).contiguous()

        x_forward = self.conv_transpose(x_forward)
        x_reverse = self.conv_transpose(x_reverse)
        return x_forward + x_reverse

def make_patch_embed(x, patch_size):
    x = x.permute(0, 2, 3, 1).contiguous()
    B, H, W, C = x.shape
    x = x.view(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    patchs = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, patch_size, patch_size, C)
    return patchs

def unpatchify(patch, B, patch_size, H, W):
    # B = int(patch.shape[0] / (H * W / patch_size / patch_size))
    x = patch.view(B, H // patch_size, W // patch_size, patch_size, patch_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    return x
class WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, res_mode='layer_res'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode
        # parameter of row cell
        self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
        self.W_other_layer = torch.nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size))
        self.B = torch.nn.Parameter(torch.empty(num_layers, 6 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def linear(self, input, weight, bias, batch_size, slice, Water2sea_slice_num):
        a = F.linear(input, weight)
        if slice < Water2sea_slice_num:
            a[:batch_size * (slice + 1), :] = a[:batch_size * (slice + 1), :] + bias
        return a

    def forward(self, input, batch_size, input_size, flag):
        if flag == 1:  # cols > rows
            input = input.permute(2, 0, 1, 3)
        else:
            input = input.permute(1, 0, 2, 3)
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device)
        for r in range(Water2sea_slice_num):
            input_transfer[r, :, r:r + Original_slice_len, :] = input[r, :, :, :]
        hidden_row_all_list = []
        hidden_col_all_list = []
        for layer in range(self.num_layers):
            if layer == 0:
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, input_size)
                W = self.W_first_layer
            else:
                a = F.dropout(output_all_slice, self.dropout, self.training)
                if layer == 1:
                    layer0_output = a
                W = self.W_other_layer[layer - 1, :, :]
                hidden_slice_row = hidden_slice_row * 0
                hidden_slice_col = hidden_slice_col * 0
            B = self.B[layer, :]
            # start every for all slice
            output_all_slice_list = []
            for slice in range(Water2sea_slice_len):
                # gate generate
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]],
                                             dim=-1), W, B, batch_size, slice, Water2sea_slice_num)
                # gate
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.hidden_size, dim=-1)
                sigmod_gate = torch.sigmoid(sigmod_gate)
                tanh_gate = torch.tanh(tanh_gate)
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim=-1)
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim=-1)
                # gate effect
                hidden_slice_row = torch.tanh(
                    (1 - update_gate_row) * hidden_slice_row + update_gate_row * input_gate_row) * output_gate_row
                hidden_slice_col = torch.tanh(
                    (1 - update_gate_col) * hidden_slice_col + update_gate_col * input_gate_col) * output_gate_col
                # output generate
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim=-1)
                # save output
                output_all_slice_list.append(output_slice)
                # save row hidden
                if slice >= Original_slice_len - 1:
                    need_save_row_loc = slice - Original_slice_len + 1
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc * batch_size:(need_save_row_loc + 1) * batch_size, :])
                # save col hidden
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num - 1) * batch_size:, :])
                # hidden transfer
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims=0)
            if self.res_mode == 'layer_res' and layer >= 1:  # layer-res
                output_all_slice = torch.stack(output_all_slice_list, dim=1) + layer0_output
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim=1)
        hidden_row_all = torch.stack(hidden_row_all_list, dim=1)
        hidden_col_all = torch.stack(hidden_col_all_list, dim=1)
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num,
                                                hidden_row_all.shape[-1])
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len,
                                                hidden_col_all.shape[-1])
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all
class WITRAN(nn.Module):
    def __init__(self, config):
        super(WITRAN, self).__init__()
        self.standard_batch_size = config.batch_size
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.dec_in = config.dec_in
        self.c_out = config.c_out
        self.d_model = config.d_model
        self.num_layers = config.e_layers
        self.dropout = config.dropout
        self.WITRAN_dec = config.WITRAN_dec
        self.WITRAN_deal = config.WITRAN_deal
        self.WITRAN_res = config.WITRAN_res
        self.PE_way = config.WITRAN_PE
        self.WITRAN_grid_cols = config.WITRAN_grid_cols
        self.WITRAN_grid_enc_rows = int(config.seq_len / self.WITRAN_grid_cols)
        self.WITRAN_grid_dec_rows = int(config.pred_len / self.WITRAN_grid_cols)
        self.device = config.gpu

        # Encoder
        self.encoder_2d = WITRAN_2DPSGMU_Encoder(self.enc_in, self.d_model, self.num_layers,
                                                 self.dropout, self.WITRAN_grid_enc_rows, self.WITRAN_grid_cols,
                                                 self.WITRAN_res)
        # Embedding

        if self.PE_way == 'add':
            if self.WITRAN_dec == 'FC':
                self.fc_1 = nn.Linear(
                    self.num_layers * (self.WITRAN_grid_enc_rows + self.WITRAN_grid_cols) * self.d_model,
                    self.pred_len * self.d_model)
            elif self.WITRAN_dec == 'Concat':
                self.fc_1 = nn.Linear(self.num_layers * 2 * self.d_model, self.WITRAN_grid_dec_rows * self.d_model)
            self.fc_2 = nn.Linear(self.d_model, self.c_out)
        else:
            if self.WITRAN_dec == 'FC':
                self.fc_1 = nn.Linear(
                    self.num_layers * (self.WITRAN_grid_enc_rows + self.WITRAN_grid_cols) * self.d_model,
                    self.pred_len * self.d_model)
            elif self.WITRAN_dec == 'Concat':
                self.fc_1 = nn.Linear(self.num_layers * 2 * self.d_model, self.WITRAN_grid_dec_rows * self.d_model)
            self.fc_2 = nn.Linear(self.d_model * 2, self.c_out)

    def forward(self, input):
        # input = input.permute(0, 2, 3, 1).contiguous()
        default_batch_size, H, W, input_size = input.shape
        seqence_len = H * W
        x_input_enc = input.reshape(-1, self.WITRAN_grid_enc_rows, self.WITRAN_grid_cols, input_size)

        batch_size = x_input_enc.shape[0]

        if self.WITRAN_grid_enc_rows <= self.WITRAN_grid_cols:
            flag = 0
        else:  # need permute
            flag = 1

        _, enc_hid_row, enc_hid_col = self.encoder_2d(x_input_enc, batch_size, input_size, flag)
        # enc_hid_row = enc_hid_row[:, :, -1:, :].expand(-1, -1, self.WITRAN_grid_cols, -1)
        output = torch.cat([enc_hid_row, enc_hid_col], dim=-1).permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], output.shape[2] * output.shape[3])
        last_output = self.fc_1(output)
        last_output = last_output.reshape(last_output.shape[0], last_output.shape[1], self.WITRAN_grid_dec_rows,
                                          self.d_model).permute(0, 2, 1, 3)
        last_output = last_output.reshape(last_output.shape[0], last_output.shape[1] * last_output.shape[2],
                                          last_output.shape[3])
        last_output = self.fc_2(last_output)
        # last_output = last_output.reshape(default_batch_size, -1, input_size)
        return last_output.reshape(-1, H, W, input_size)
class Config():
    def __init__(self, c_in=32, c_out=32, patch_size=4, d_model=64, WITRAN_grid_cols=16, pred_len=256, seq_len=256):
        # super(Config1, self).__init__()
        self.batch_size = 1
        self.pred_len = pred_len
        self.enc_in = c_in  # 输入维度
        self.patch_size = patch_size
        self.dec_in = 1
        self.c_out = c_out  # 输出维度
        self.d_model = d_model
        self.e_layers = 3
        self.dropout = 0.05
        self.WITRAN_deal = 'None'
        self.gpu = 0
        self.WITRAN_grid_cols = WITRAN_grid_cols
        self.seq_len = seq_len
        self.WITRAN_dec = 'Concat'
        self.WITRAN_res = 'layer_res'
        self.WITRAN_PE = 'add'
        self.is_light_sr = False
class WITBLOCK(nn.Module):
    def __init__(self, c_in, c_out, patch_size, d_model, WITRAN_grid_cols, pred_len, seq_len):
        super(WITBLOCK, self).__init__()
        self.input_dim = c_in
        self.out_dim = c_out
        self.patch_size = patch_size
        self.d_model = d_model
        self.PatchEmbed = PatchEmbed(in_channels=c_in, patch_size=patch_size, emb_size=c_out)
        self.unpatch = Unpatchify(in_channels=c_in, patch_size=patch_size, emb_size=c_out)
        self.is_light_sr = "False"
        self.d_conv = 3,
        self.conv_bias = True,
        self.bias = False,
        self.dtype = None,
        self.in_norm = nn.LayerNorm(self.input_dim)
        self.out_norm = nn.LayerNorm(self.input_dim)
        self.norm3 = nn.LayerNorm(self.out_dim)
        self.conv_blk = CAB(self.d_model // 2, self.is_light_sr)
        self.ln_2 = nn.LayerNorm(self.d_model // 2)
        self.skip_scale1 = nn.Parameter(torch.ones(self.d_model // 2))
        self.skip_scale2 = nn.Parameter(torch.ones(self.d_model // 2))
        self.in_proj = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.out_proj = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.conv1d = CausalConv1d(dim=self.input_dim, kernel_size=4, bias=True, )
        # self.conv2d = nn.Conv2d(in_channels=self.input_dim, out_channels=self.input_dim, groups=self.input_dim,
        #                         bias=False, kernel_size=3,  # padding=(self.d_conv - 1) // 2,
        #                         padding=1, )
        self.conv2d = nn.Conv2d(in_channels=self.input_dim, out_channels=self.input_dim, groups=self.input_dim,
                                bias=False, kernel_size=3,  # padding=(self.d_conv - 1) // 2,
                                padding=1, )
        # self.dropout = nn.Dropout(self.dropout) if self.dropout > 0. else None
        self.dropout = nn.Dropout(0.05)
        self.act = nn.SiLU()
        self.WIT = WITRAN(
            Config(c_in=c_in, c_out=c_out, d_model=d_model, WITRAN_grid_cols=WITRAN_grid_cols, pred_len=pred_len,
                   seq_len=seq_len))
        self.mlp = Mlp(in_features=self.input_dim, hidden_features=self.d_model, out_features=self.out_dim,
                       act_layer=nn.GELU,
                       drop=0, channels_first=False)
        self.BatchNorm2d = nn.BatchNorm2d(1)
        self.ReLU = nn.ReLU()

    def forward(self, input):
        skip0 = input
        input = self.PatchEmbed(input)
        # B, c, h, W = input.shape
        skip = input
        input_norm = self.in_norm(input)
        input_inpro0 = self.in_proj(input_norm)
        input_inpro1 = input_inpro0.permute(0, 3, 1, 2).contiguous()
        x1 = self.conv2d(input_inpro1)
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.act(x1)
        x1_WIT = self.WIT(x1)
        x1 = self.out_norm(x1_WIT)
        x2 = self.act(input_inpro0)
        x = x1 * x2
        out = self.out_proj(x)
        out = out + skip * self.skip_scale1
        output = out * self.skip_scale2 + self.conv_blk(self.ln_2(out).permute(0, 3, 1, 2).contiguous()).permute(0, 2,
                                                                                                                 3,
                                                                                                                 1).contiguous()
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.unpatch(output)
        return output + skip0
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv0_0 = Block(64, 32)
        self.conv0_1 = Block(32, 16)

        self.conv00 = Block(1, 16)
        self.conv10 = Block(1, 16)
        self.conv20 = Block(1, 16)

        self.conv01 = Block(16, 16)
        self.conv02 = Block(16, 32)
        self.conv03 = Block(32, 64)
        self.conv04 = Block(64, 128)

        self.conv11 = Block(16, 16)
        self.conv12 = Block(16, 32)
        self.conv13 = Block(32, 64)
        self.conv14 = Block(64, 128)

        self.conv21 = Block(16, 16)
        self.conv22 = Block(16, 32)
        self.conv23 = Block(32, 64)
        self.conv24 = Block(64, 128)

        config01 = Config(c_in=16, c_out=16, patch_size=4, d_model=32, WITRAN_grid_cols=16, pred_len=256, seq_len=256)
        config02 = Config(c_in=16, c_out=16, patch_size=4, d_model=32, WITRAN_grid_cols=16, pred_len=256, seq_len=256)
        config03 = Config(c_in=32, c_out=32, patch_size=4, d_model=64, WITRAN_grid_cols=16, pred_len=256, seq_len=256)
        config04 = Config(c_in=64, c_out=64, patch_size=4, d_model=128, WITRAN_grid_cols=16, pred_len=256, seq_len=256)

        config11 = Config(c_in=16, c_out=16, patch_size=4, d_model=32, WITRAN_grid_cols=8, pred_len=64, seq_len=64)
        config12 = Config(c_in=16, c_out=16, patch_size=4, d_model=32, WITRAN_grid_cols=8, pred_len=64, seq_len=64)
        config13 = Config(c_in=32, c_out=32, patch_size=4, d_model=64, WITRAN_grid_cols=8, pred_len=64, seq_len=64)
        config14 = Config(c_in=64, c_out=64, patch_size=4, d_model=128, WITRAN_grid_cols=8, pred_len=64, seq_len=64)

        config21 = Config(c_in=16, c_out=16, patch_size=4, d_model=32, WITRAN_grid_cols=4, pred_len=16, seq_len=16)
        config22 = Config(c_in=16, c_out=16, patch_size=4, d_model=32, WITRAN_grid_cols=4, pred_len=16, seq_len=16)
        config23 = Config(c_in=32, c_out=32, patch_size=4, d_model=64, WITRAN_grid_cols=4, pred_len=16, seq_len=16)
        config24 = Config(c_in=64, c_out=64, patch_size=4, d_model=128, WITRAN_grid_cols=4, pred_len=16, seq_len=16)

        self.WIT01 = WITBLOCK(config01.enc_in, config01.c_out, config01.patch_size, config01.d_model,
                              config01.WITRAN_grid_cols, config01.pred_len, config01.pred_len)
        self.WIT02 = WITBLOCK(config02.enc_in, config02.c_out, config02.patch_size, config02.d_model,
                              config02.WITRAN_grid_cols, config02.pred_len, config02.pred_len)
        self.WIT03 = WITBLOCK(config03.enc_in, config03.c_out, config03.patch_size, config03.d_model,
                              config03.WITRAN_grid_cols, config03.pred_len, config03.pred_len)
        self.WIT04 = WITBLOCK(config04.enc_in, config04.c_out, config04.patch_size, config04.d_model,
                              config04.WITRAN_grid_cols, config04.pred_len, config04.pred_len)

        self.WIT11 = WITBLOCK(config11.enc_in, config11.c_out, config11.patch_size, config11.d_model,
                              config11.WITRAN_grid_cols, config11.pred_len, config11.pred_len)
        self.WIT12 = WITBLOCK(config12.enc_in, config12.c_out, config12.patch_size, config12.d_model,
                              config12.WITRAN_grid_cols, config12.pred_len, config12.pred_len)
        self.WIT13 = WITBLOCK(config13.enc_in, config13.c_out, config13.patch_size, config13.d_model,
                              config13.WITRAN_grid_cols, config13.pred_len, config13.pred_len)
        self.WIT14 = WITBLOCK(config14.enc_in, config14.c_out, config14.patch_size, config14.d_model,
                              config14.WITRAN_grid_cols, config14.pred_len, config14.pred_len)

        self.WIT21 = WITBLOCK(config21.enc_in, config21.c_out, config21.patch_size, config21.d_model,
                              config21.WITRAN_grid_cols, config21.pred_len, config21.pred_len)
        self.WIT22 = WITBLOCK(config22.enc_in, config22.c_out, config22.patch_size, config22.d_model,
                              config22.WITRAN_grid_cols, config22.pred_len, config22.pred_len)
        self.WIT23 = WITBLOCK(config23.enc_in, config23.c_out, config23.patch_size, config23.d_model,
                              config23.WITRAN_grid_cols, config23.pred_len, config23.pred_len)
        self.WIT24 = WITBLOCK(config24.enc_in, config24.c_out, config24.patch_size, config24.d_model,
                              config24.WITRAN_grid_cols, config24.pred_len, config24.pred_len)

        self.down01 = DownSamplingBlock(1, 1)
        self.down02 = DownSamplingBlock(16, 16)
        self.down03 = DownSamplingBlock(32, 32)
        self.down04 = DownSamplingBlock(64, 64)
        self.down11 = DownSamplingBlock(1, 1)
        self.down12 = DownSamplingBlock(16, 16)
        self.down13 = DownSamplingBlock(32, 32)
        self.down14 = DownSamplingBlock(64, 64)

        self.up00 = UpSamplingBlock(1, 1)
        self.up10 = UpSamplingBlock(1, 1)
        self.up01 = UpSamplingBlock(64, 64)
        self.up11 = UpSamplingBlock(64, 64)

        self.up0 = UpSamplingBlock(256, 128)
        self.up1 = UpSamplingBlock(128, 64)
        self.up2 = UpSamplingBlock(64, 64)
        self.up3 = UpSamplingBlock(32, 16)

        self.up_0 = UpSamplingBlock(64, 64)

        self.conv1 = Block(1, 3)
        self.conv2 = Block(1, 3)

        self.down_conv0 = UpSamplingBlock(512, 256)
        self.down_conv1 = UpSamplingBlock(256, 128)
        self.down_conv2 = UpSamplingBlock(128, 64)
        self.down_conv3 = UpSamplingBlock(64, 64)
        self.down_conv4 = Block(64, 32)
        self.down_conv5 = Block(32, 16)

        self.DENetFeatureExtractor = DENetFeatureExtractor()


        self.LSTM10 = GSC_batch_2(16, 16, 1, 16, 16)
        self.LSTM11 = GSC_batch_2(32, 32, 1, 16, 16)
        self.LSTM12 = GSC_batch_2(64, 64, 1, 16, 16)

        self.LSTM20 = GSC_batch_2(16, 16, 1, 16, 16)
        self.LSTM21 = GSC_batch_2(32, 32, 1, 16, 16)
        self.LSTM22 = GSC_batch_2(64, 64, 1, 16, 16)

    def forward(self, input1, input2):

##############自提示信息生成##########################
        P_ir = input1.repeat(1, 3, 1, 1)
        P_vis = input2.repeat(1, 3, 1, 1)
        #
        load_seg = LoadSegFeature(input1)

        ir_img = input1
        vis_img = input2

        f_P_ir_256 = load_seg.getIrFeature3()   #(B, 256, H/8, W/8)
        f_P_ir_128 = self.up0(f_P_ir_256)        #(B, 126, H/4, W/4)
        f_P_ir_64 = self.up1(f_P_ir_128)         #(B, 64, H/2, W/2)
        f_P_ir_64 = self.up2(f_P_ir_64)          #(B, 64, H, W)
        f_P_ir_32 = self.conv0_0(f_P_ir_64)      #(B, 32, H, W)
        f_P_ir_16 = self.conv0_1(f_P_ir_32)      #(B, 16, H, W)

        f_P_vis_64 = self.DENetFeatureExtractor.extract_features(vis_img)

        f_P_vis_32 = self.down_conv4(f_P_vis_64)
        f_P_vis_16 = self.down_conv5(f_P_vis_32)
 ########################input1###########################
        fA11 = self.down01(input1)
        fA21 = self.down11(fA11)

        #####################原始尺寸#####################################

        fA01 = self.conv00(input1)
        fA11 = self.conv10(fA11)
        fA21 = self.conv20(fA21)

        fA02 = self.conv01(self.WIT01(fA01))
        fA02 = self.LSTM10(fA02, f_P_ir_16, 16) + fA02
        fA03 = self.conv02(self.WIT02(fA02))
        fA03 = self.LSTM11(fA03, f_P_ir_32, 16) + fA03
        fA04 = self.conv03(self.WIT03(fA03))
        fA04 = self.LSTM12(fA04, f_P_ir_64, 16) + fA04
        #####################1/2原始尺寸#####################################
        fA12 = self.conv11(self.WIT11(fA11))
        tpA11 = fA12 + self.down02(fA02)
        fA13 = self.conv12(self.WIT12(tpA11))
        tpA12 = fA13 + self.down03(fA03)
        fA14 = self.conv13(self.WIT13(tpA12))
        tpA13 = fA14 + self.down04(fA04)
        #####################1/4原始尺寸#####################################
        fA22 = self.conv21(self.WIT21(fA21))
        tpA21 = fA22 + self.down12(tpA11)
        fA23 = self.conv22(self.WIT22(tpA21))
        tpA22 = fA23 + self.down13(tpA12)
        fA24 = self.conv23(self.WIT23(tpA22))
        tpA23 = fA24 + self.down14(tpA13)
        ##########################################################
        tpA14 = tpA13 + self.up11(tpA23)
        tpA04 = fA04 + self.up01(tpA14)

 ########################input2###########################
        fB11 = self.down01(input2)
        fB21 = self.down11(fB11)

        fB01 = self.conv00(input2)
        fB11 = self.conv10(fB11)
        fB21 = self.conv20(fB21)

        fB02 = self.conv01(fB01)
        fB02 = self.LSTM20(fB02, f_P_vis_16, 16) + fB02
        fB03 = self.conv02(self.WIT02(fB02))
        fB03 = self.LSTM21(fB03, f_P_vis_32, 16) + fB03
        fB04 = self.conv03(self.WIT03(fB03))
        fB04 = self.LSTM22(fB04, f_P_vis_64, 16) + fB04
        #####################1/2原始尺寸#####################################
        fB12 = self.conv11(self.WIT11(fB11))
        tpB11 = fB12 + self.down02(fB02)
        fB13 = self.conv12(self.WIT12(tpB11))
        tpB12 = fB13 + self.down03(fB03)
        fB14 = self.conv13(self.WIT13(tpB12))
        tpB13 = fB14 + self.down04(fB04)
        #####################1/4原始尺寸#####################################
        fB22 = self.conv21(self.WIT21(fB21))
        tpB21 = fB22 + self.down12(tpB11)
        fB23 = self.conv22(self.WIT22(tpB21))
        tpB22 = fB23 + self.down13(tpB12)
        fB24 = self.conv23(self.WIT23(tpB22))
        tpB23 = fB24 + self.down14(tpB13)
        ##########################################################
        tpB14 = tpB13 + self.up11(tpB23)
        tpB04 = fB04 + self.up01(tpB14)


        return tpA04, tpB04, fA04, fA14, fA24, fB04, fB14, fB24, f_P_ir_16, f_P_ir_32, f_P_ir_64, f_P_vis_16 ,f_P_vis_32, f_P_vis_64

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )

        self.decoder5 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, tpA04, tpB04, f_P_ir_64, f_P_vis_64):
        output1 = self.decoder1(tpA04)
        output2 = self.decoder2(tpB04)
        output3 = self.decoder3(f_P_ir_64)
        output4 = self.decoder4(f_P_vis_64)

        return output1, output2, output3, output4
