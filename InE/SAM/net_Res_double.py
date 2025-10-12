import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

from SAM.layers import ConvLeakyRelu2d
from SAM.irnn import irnn


class Spacial_IRNN(nn.Module):
    def __init__(self, in_channels, alpha=0.2):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.IRNN = irnn()

    def forward(self, input):
        output = self.IRNN.apply(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight,
                                 self.left_weight.weight, self.up_weight.bias, self.right_weight.bias,
                                 self.down_weight.bias,
                                 self.left_weight.bias)
        return output


class Attention_SAM(nn.Module):
    def __init__(self, in_channels):
        super(Attention_SAM, self).__init__()
        model = []
        out_channels = int(in_channels / 2)
        model += [ConvLeakyRelu2d(in_channels, out_channels)]
        model += [ConvLeakyRelu2d(out_channels, out_channels)]
        model += [ConvLeakyRelu2d(out_channels, 4, activation='Sigmod', kernel_size=3, padding=1, stride=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels)
        self.irnn2 = Spacial_IRNN(self.out_channels)
        self.conv_in = ConvLeakyRelu2d(in_channels, in_channels, activation=None)
        self.conv2 = ConvLeakyRelu2d(in_channels * 4, in_channels, activation=None, kernel_size=3, padding=1, stride=1)
        self.conv3 = ConvLeakyRelu2d(in_channels * 4, in_channels, kernel_size=3, padding=1, stride=1)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention_SAM(in_channels)
        self.conv_out = ConvLeakyRelu2d(in_channels, in_channels, activation='Sigmod', kernel_size=3, padding=1,
                                        stride=1)

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            # print('top_up device:', top_up.device, 'weight device:', weight.device)
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        mask = self.conv_out(out)
        return mask

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.patch_embed_vi = OverlapPatchEmbed(1, 64)

        self.encoder_level_vi = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(4)])
        self.patch_embed_ir = OverlapPatchEmbed(1, 64)

        self.encoder_level_ir = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=2,
                               bias=False, LayerNorm_type='WithBias') for i in range(4)])

    def forward(self, x, ir):
        x = self.patch_embed_vi(x)
        x = self.encoder_level_vi(x)

        ir = self.patch_embed_ir(ir)
        ir = self.encoder_level_ir(ir)
        return x, ir


class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=112, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=112, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.last_ir = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1)
        self.last_vi = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1)

        self.rule = nn.LeakyReLU()
        self.tan = nn.Tanh()

    def forward(self, x, ir, stage=True):
        if stage:
            x1 = self.rule(self.conv1(x))
            x2 = self.rule(self.conv2(torch.cat((x1, x), dim=1)))
            x3 = self.rule(self.conv3(torch.cat((x2, x1, x), dim=1)))
            x4 = self.conv4(x3)
            ir1 = self.rule(self.conv5(ir))
            ir2 = self.rule(self.conv6(torch.cat((ir1, ir), dim=1)))
            ir3 = self.rule(self.conv7(torch.cat((ir2, ir1, ir), dim=1)))
            ir4 = self.conv8(ir3)
            return self.tan(self.last_vi(x4)), self.tan(self.last_ir(ir4))
        else:
            x1 = self.rule(self.conv5(x))
            x2 = self.rule(self.conv6(torch.cat((x1, x), dim=1)))
            x3 = self.rule(self.conv7(torch.cat((x2, x1, x), dim=1)))
            x4 = self.conv8(x3)

            ir1 = self.rule(self.conv1(ir))
            ir2 = self.rule(self.conv2(torch.cat((ir1, ir), dim=1)))
            ir3 = self.rule(self.conv3(torch.cat((ir2, ir1, ir), dim=1)))
            ir4 = self.conv4(ir3)

            return self.tan(self.last_ir(x4)), self.tan(self.last_vi(ir4))

# class Decode(nn.Module):
#     def __init__(self):
#         super(Decode, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=112, out_channels=8, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
#
#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv6 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv7 = nn.Conv2d(in_channels=112, out_channels=8, kernel_size=3, stride=1, padding=1)
#         self.conv8 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
#
#         self.rule = nn.LeakyReLU()
#         self.tan = nn.Tanh()
#
#     def forward(self, x, ir, stage=True):
#         if stage:
#             x1 = self.rule(self.conv1(x))
#             x2 = self.rule(self.conv2(torch.cat((x1, x), dim=1)))
#             x3 = self.rule(self.conv3(torch.cat((x2, x1, x), dim=1)))
#             x4 = self.conv4(x3)
#             ir1 = self.rule(self.conv5(ir))
#             ir2 = self.rule(self.conv6(torch.cat((ir1, ir), dim=1)))
#             ir3 = self.rule(self.conv7(torch.cat((ir2, ir1, ir), dim=1)))
#             ir4 = self.conv8(ir3)
#             return self.tan(x4), self.tan(ir4)
#         else:
#             x1 = self.rule(self.conv5(x))
#             x2 = self.rule(self.conv6(torch.cat((x1, x), dim=1)))
#             x3 = self.rule(self.conv7(torch.cat((x2, x1, x), dim=1)))
#             x4 = self.conv8(x3)
#
#             ir1 = self.rule(self.conv1(ir))
#             ir2 = self.rule(self.conv2(torch.cat((ir1, ir), dim=1)))
#             ir3 = self.rule(self.conv3(torch.cat((ir2, ir1, ir), dim=1)))
#             ir4 = self.conv4(ir3)
#
#             return self.tan(x4), self.tan(ir4)


class Common(nn.Module):
    def __init__(self):
        super(Common, self).__init__()
        self.sam1 = SAM(64, 64, 1)
        self.sam2 = SAM(64, 64, 1)

    def forward(self, vi, ir):
        vi_SAM = self.sam1(vi)
        feature_vi = vi_SAM.mul(vi)

        ir_SAM = self.sam2(ir)
        feature_ir = ir_SAM.mul(ir)

        return feature_vi, feature_ir


if __name__ == '__main__':
    a = torch.ones([1, 1, 240, 240]).cuda()
    c = torch.randn([1, 1, 240, 240]).cuda()
    net = Encode().cuda()
    net2 = Common().cuda()
    net3 = Decode().cuda()
    b, d = net(a, c)
    b, d = net2(b, d)
    b, d = net3(b, d)
    print(d)