import torch
import torch.nn as nn
import numpy as np

from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act

from .ffc import FFC_BN_ACT, FFCResnetBlock, ConcatTupleLayer
from .base import get_activation
from .spatial_transform import LearnableSpatialTransformWrapper
from .ffc_style_module import DecBlock, FFCResnetBlock_Style, LearnableSpatialTransformWrapper_Style


class FFCStyleResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={},
                 refine=False, input_mirror=None, cat_inv=True, use_style="style_res", **kwargs):
        assert (n_blocks >= 0)
        super().__init__()

        print("Use FFCStyleResNetGenerator!")

        # input_nc = 0
        self.input_mirror = input_mirror

        if self.input_mirror == "cat":
            if cat_inv:
                input_nc = (3 + 3 + 1) + (3 + 1) # (img + inversion + mask) + (img_mirror + mask_mirror) = 11
            else:
                input_nc = (3 + 1) + (3 + 1)
        elif self.input_mirror == "condition_twice":
            input_nc = 3 + 1 # img + mask
        else:
            if cat_inv:
                input_nc = 3 + 3 + 1  # img + inversion + mask = 7
            else:
                input_nc = 3 + 1
        self.input_nc = input_nc
        print("input_nc:", self.input_nc)

        self.use_style = use_style
        print("Use Style:", self.use_style)


        ### downsample
        model_down = [nn.ReflectionPad2d(3),
                      FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                                 activation_layer=activation_layer, **init_conv_kwargs)]
        for i in range(n_downsampling):  # 512 -> 256 -> 128 -> 64
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model_down += [FFC_BN_ACT(min(max_features, ngf * mult),
                                      min(max_features, ngf * mult * 2),
                                      kernel_size=3, stride=2, padding=1,
                                      norm_layer=norm_layer,
                                      activation_layer=activation_layer,
                                      **cur_conv_kwargs)]
        # model_down += [ConcatTupleLayer()]  #### 128, 384 -> 512
        self.model_down = nn.Sequential(*model_down)

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)  # 512


        if self.input_mirror == "condition":
            cond_conv_kwargs = dict(resnet_conv_kwargs)
            self.cond = CondEncoder(feats_num_bottleneck, norm_layer, activation_layer, cond_conv_kwargs)
        elif self.input_mirror == "condition_twice":
            cond_conv_kwargs = dict(resnet_conv_kwargs)
            self.cond1 = CondEncoder(feats_num_bottleneck, norm_layer, activation_layer, cond_conv_kwargs)
            self.cond2 = CondEncoder(feats_num_bottleneck, norm_layer, activation_layer, cond_conv_kwargs)

        ### style
        self.style_dim = 512

        ### resnet blocks
        model_res = []
        for i in range(n_blocks):
            if self.use_style == "style_res":
                cur_resblock = FFCResnetBlock_Style(feats_num_bottleneck, padding_type=padding_type,
                                                    activation_layer=activation_layer, norm_layer=norm_layer,
                                                    style_dim=self.style_dim, res=6, use_noise=False, activation='lrelu', demodulate=True,
                                                    **resnet_conv_kwargs)
                if spatial_transform_layers is not None and i in spatial_transform_layers:
                    cur_resblock = LearnableSpatialTransformWrapper_Style(cur_resblock, **spatial_transform_kwargs)

            else:
                cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                              activation_layer=activation_layer, norm_layer=norm_layer,
                                              **resnet_conv_kwargs)  # , inline=True
                if spatial_transform_layers is not None and i in spatial_transform_layers:
                    cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
                    
            model_res += [cur_resblock]

        # model_res += [ConcatTupleLayer()]  #### 128, 384 -> 512
        self.model_res = nn.Sequential(*model_res)

        self.concat = ConcatTupleLayer()

        
        ############################################################################################################

        ### upsample
        model_up = []
        res = 64
        for i in range(n_downsampling):  # 64 -> 128 -> 256 -> 512
            mult = 2 ** (n_downsampling - i)
            res = res * 2

            if self.use_style is not None:
                model_up += [DecBlock(res, min(max_features, ngf * mult), min(max_features, int(ngf * mult / 2)),
                                      activation='lrelu', style_dim=self.style_dim,
                                      use_noise=False, demodulate=True, img_channels=3)]  # img_channels = 3 or ngf
            else:
                model_up += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                                min(max_features, int(ngf * mult / 2)),
                                                kernel_size=3, stride=2, padding=1, output_padding=1),
                             up_norm_layer(min(max_features, int(ngf * mult / 2))),
                             up_activation]

        self.model_up = nn.Sequential(*model_up)
        ############################################################################################################


        ### output
        model_out = []
        if out_ffc:
            if self.use_style is not None:
                model_out += [FFCResnetBlock_Style(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                                   norm_layer=norm_layer, inline=True,
                                                   style_dim=self.style_dim, res=6, use_noise=False, activation='lrelu', demodulate=True,
                                                   **out_ffc_kwargs)]
            else:
                model_out += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                             norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model_out += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model_out.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model_out = nn.Sequential(*model_out)

        self.refine = refine
        if self.use_style is not None and self.refine:
            self.refinenet = RefineNet(in_ch=input_nc+3, res=512, style_dim=self.style_dim,
                                       activation='lrelu', use_noise=False, demodulate=True, img_channels=3)


    def forward(self, inp, ws):
        
        if self.input_mirror == "condition":
            inp_ori, inp_mirror = torch.split(inp, self.input_nc, dim=1)
            x_ori = self.model_down(inp_ori)
            x_mirror = self.model_down(inp_mirror)
            x = self.cond(x_ori, x_mirror)
        elif self.input_mirror == "condition_twice":
            inp_ori, inp_inv, inp_mirror = torch.split(inp, self.input_nc, dim=1)
            x_ori = self.model_down(inp_ori)
            x_inv = self.model_down(inp_inv)
            x_mirror = self.model_down(inp_mirror)
            x = self.cond1(x_inv, x_mirror)
            x = self.cond2(x_ori, x)
        else:
            x = self.model_down(inp)

        if self.use_style == "style_res":
            for block in self.model_res:
                x = block(x, ws[:, -5:-4])
        else:
            x = self.model_res(x)
        
        x = self.concat(x)

        if self.use_style is not None:
            ws_up = torch.cat([ws[:, -5:], ws[:, -1:].repeat(1, 2, 1)], dim=1)
            img = None
            for i, block in enumerate(self.model_up):
                x, img = block(x, img, ws_up.narrow(1, i * 2, 3))
            
            out = self.model_out(x)
            
            # refine
            if self.refine:
                inp_refine = torch.cat([out, inp], dim=1)
                # inp_refine = torch.cat([masked_img, inversion, mask], dim=1)
                ws_refine = ws[:, -10:]
                ws_refine = torch.cat([ws_refine, ws_refine[:, -1:].repeat(1, 2, 1)], dim=1)
                out = self.refinenet(inp_refine, ws_refine)
        else:
            x = self.model_up(x)
            out = self.model_out(x)

        return out


class CondEncoder(nn.Module):
    def __init__(self, channel, norm_layer, activation_layer, cur_conv_kwargs):
        super().__init__()

        self.condition_scale = nn.Sequential(
            FFC_BN_ACT(2 * channel, channel, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer, activation_layer=activation_layer, **cur_conv_kwargs),
            FFC_BN_ACT(channel, channel, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer, activation_layer=activation_layer, **cur_conv_kwargs)
        )
        self.condition_shift = nn.Sequential(
            FFC_BN_ACT(2 * channel, channel, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer, activation_layer=activation_layer, **cur_conv_kwargs),
            FFC_BN_ACT(channel, channel, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer, activation_layer=activation_layer, **cur_conv_kwargs)
        )
    
    def forward(self, x_ori, x_mirror):
        x_ori_l, x_ori_g = x_ori
        x_mirror_l, x_mirror_g = x_mirror
        x_cond_l = torch.cat([x_ori_l, x_mirror_l], dim=1)
        x_cond_g = torch.cat([x_ori_g, x_mirror_g], dim=1)
        x_cond = (x_cond_l, x_cond_g)

        scale = self.condition_scale(x_cond)
        scale_l, scale_g = scale
        shift = self.condition_shift(x_cond)
        shift_l, shift_g = shift

        x_ori_l = x_ori_l * (1 + scale_l) + shift_l
        x_ori_g = x_ori_g * (1 + scale_g) + shift_g
        x = (x_ori_l, x_ori_g)

        return x
