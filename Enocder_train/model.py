import os
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import math
from module_attention_1 import ModifiedSpatialTransformer
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class InE_enocder(nn.Module):
    def __init__(self, input_nc, ndf=64, semantic_dim=36, semantic_size=16, use_bias=True, nheads=1, dhead=64):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()

        kw = 4
        padw = 1
        # ss = [128, 64, 32, 31, 30]  # PatchGAN's spatial size
        # cs = [64, 128, 256, 512, 1]  # PatchGAN's channel size

        norm = spectral_norm

        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv_first = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1)
        self.seg_convfirst = norm(nn.Conv2d(9, 18, kernel_size=kw, stride=2, padding=1, bias=use_bias))

        self.conv1 = norm(nn.Conv2d(ndf * 1, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.seg_conv1 = norm(nn.Conv2d(18, 36, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = 1
        self.att1 = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=128,
                                               up_factor=upscale)

        ex_ndf = semantic_dim
        self.conv11 = norm(nn.Conv2d(ndf * 2 + ex_ndf, ndf * 2, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.seg_conv2 = norm(nn.Conv2d(36, 72, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = 1
        self.att2 = ModifiedSpatialTransformer(in_channels=semantic_dim*2, n_heads=nheads, d_head=dhead, context_dim=256,
                                               up_factor=upscale)

        ex_ndf = semantic_dim * 2
        self.conv21 = norm(nn.Conv2d(ndf * 4 + ex_ndf, ndf * 4, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.seg_conv3 = norm(nn.Conv2d(72, 144, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        upscale = 1
        self.att3 = ModifiedSpatialTransformer(in_channels=semantic_dim*4, n_heads=nheads, d_head=dhead, context_dim=512,
                                               up_factor=upscale, is_last=True)

        ex_ndf = semantic_dim * 4
        self.conv31 = norm(nn.Conv2d(ndf * 8 + ex_ndf, ndf * 8, kernel_size=3, stride=1, padding=padw, bias=use_bias))

        self.decode_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=use_bias)
        self.decode_2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=use_bias)
        self.decode_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.decode_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.decode_5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=use_bias)

        init_weights(self, init_type='normal')

    def forward(self, input, semantic):
        """Standard forward."""
        input = self.conv_first(input)
        input = self.lrelu(input)
        input = self.conv1(input)
        semantic = self.seg_convfirst(semantic)
        semantic = self.lrelu(semantic)
        semantic = self.seg_conv1(semantic)
        se = self.att1(semantic, input)
        input = self.lrelu(self.conv11(torch.cat([input, se], dim=1)))

        input = self.conv2(input)
        semantic = self.seg_conv2(semantic)
        se = self.att2(semantic, input)
        input = self.lrelu(self.conv21(torch.cat([input, se], dim=1)))

        input = self.conv3(input)
        semantic = self.seg_conv3(semantic)
        se = self.att3(semantic, input)
        input = self.lrelu(self.conv31(torch.cat([input, se], dim=1)))
        feature = input
        # if self.training:
            # 如果在训练模式下，执行解码操作
        input = self.lrelu(self.decode_1(input))
        input = self.lrelu(self.decode_2(input))
        input = self.lrelu(self.decode_3(input))
        input = self.lrelu(self.decode_4(input))
        input = self.tanh(self.decode_5(input)) / 2 + 0.5
        # else:
        #     return input, feature

        return input, feature