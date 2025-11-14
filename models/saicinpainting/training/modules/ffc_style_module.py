import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import rotate

from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act

from models.saicinpainting.training.modules.spatial_transform import LearnableSpatialTransformWrapper
from models.saicinpainting.training.modules.squeeze_excitation import SELayer


class FourierUnit_Style(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho',
                 style_dim=512, res=6, use_noise=False, activation='lrelu', demodulate=True):
        # bn_layer not used
        super(FourierUnit_Style, self).__init__()
        self.groups = groups

        # self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
        #                                   out_channels=out_channels * 2,
        #                                   kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.conv_layer = StyleConv(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                    out_channels=out_channels * 2,
                                    style_dim=style_dim, resolution=2**res,
                                    kernel_size=1, use_noise=use_noise,
                                    activation=activation, demodulate=demodulate,)
        # self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        # self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x, style):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ## ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ## ffted = self.relu(self.bn(ffted))
        ffted = self.conv_layer(ffted, style)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform_Style(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True,
                 style_dim=512, res=6, use_noise=False, activation='lrelu', demodulate=True,
                 **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform_Style, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels //
        #               2, kernel_size=1, groups=groups, bias=False),
        #     nn.BatchNorm2d(out_channels // 2),
        #     nn.ReLU(inplace=True)
        # )
        self.conv1 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels // 2,
                               style_dim=style_dim, resolution=2**res,
                               kernel_size=1, use_noise=use_noise,
                               activation=activation, demodulate=demodulate,)
        self.fu = FourierUnit_Style(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs,
            style_dim=style_dim, res=res, use_noise=use_noise, activation=activation, demodulate=demodulate)
        if self.enable_lfu:
            self.lfu = FourierUnit_Style(
                out_channels // 2, out_channels // 2, groups,
                style_dim=style_dim, res=res, use_noise=use_noise, activation=activation, demodulate=demodulate)
        # self.conv2 = torch.nn.Conv2d(
        #     out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)
        self.conv2 = StyleConv(in_channels=out_channels // 2,
                               out_channels=out_channels,
                               style_dim=style_dim, resolution=2**res,
                               kernel_size=1, use_noise=use_noise,
                               activation=activation, demodulate=demodulate,)

    def forward(self, x, style):

        x = self.downsample(x)
        x = self.conv1(x, style)
        output = self.fu(x, style)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs, style)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs, style)

        return output


class FFC_Style(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False,
                 style_dim=512, res=6, use_noise=False, activation='lrelu', demodulate=True,
                 **spectral_kwargs):
        super(FFC_Style, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        # module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        # self.convl2l = module(in_cl, out_cl, kernel_size,
        #                       stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        # self.convl2g = module(in_cl, out_cg, kernel_size,
        #                       stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        # self.convg2l = module(in_cg, out_cl, kernel_size,
        #                       stride, padding, dilation, groups, bias, padding_mode=padding_type)
        # module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        # self.convg2g = module(
        #     in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        # self.gated = gated
        # module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        # self.gate = module(in_channels, 2, 1)


        module = nn.Identity if in_cl == 0 or out_cl == 0 else StyleConv
        # self.convl2l = module(in_cl, out_cl, kernel_size,
        #                       stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.convl2l = module(in_channels=in_cl, out_channels=out_cl,
                              style_dim=style_dim, resolution=2**res,
                              kernel_size=kernel_size, use_noise=use_noise,
                              activation=activation, demodulate=demodulate,)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else StyleConv
        # self.convl2g = module(in_cl, out_cg, kernel_size,
        #                       stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.convl2g = module(in_channels=in_cl, out_channels=out_cg,
                              style_dim=style_dim, resolution=2**res,
                              kernel_size=kernel_size, use_noise=use_noise,
                              activation=activation, demodulate=demodulate,)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else StyleConv
        # self.convg2l = module(in_cg, out_cl, kernel_size,
        #                       stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.convg2l = module(in_channels=in_cg, out_channels=out_cl,
                              style_dim=style_dim, resolution=2**res,
                              kernel_size=kernel_size, use_noise=use_noise,
                              activation=activation, demodulate=demodulate,)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform_Style
        # self.convg2g = module(
        #     in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs,
            style_dim=style_dim, res=res, use_noise=use_noise, activation=activation, demodulate=demodulate)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else StyleConv
        # self.gate = module(in_channels, 2, 1)
        self.gate = module(in_channels=in_channels, out_channels=2,
                              style_dim=style_dim, resolution=2**res,
                              kernel_size=1, use_noise=use_noise,
                              activation=activation, demodulate=demodulate,)
        

    def forward(self, x, style):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input, style))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l, style) + self.convg2l(x_g, style) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l, style) * l2g_gate + self.convg2g(x_g, style)

        return out_xl, out_xg


class FFC_BN_ACT_Style(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True,
                 style_dim=512, res=6, use_noise=False, activation='lrelu', demodulate=True,
                 **kwargs):
        super(FFC_BN_ACT_Style, self).__init__()
        self.ffc = FFC_Style(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type,
                       style_dim=style_dim, res=res, use_noise=use_noise, activation=activation, demodulate=demodulate,
                       **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x, style):
        x_l, x_g = self.ffc(x, style)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class LearnableSpatialTransformWrapper_Style(nn.Module):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x, style):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x), style), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans, style)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y


class FFCResnetBlock_Style(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False,
                 style_dim=512, res=6, use_noise=False, activation='lrelu', demodulate=True,
                 **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT_Style(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                style_dim=style_dim, res=res, use_noise=use_noise, activation=activation, demodulate=demodulate,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT_Style(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                style_dim=style_dim, res=res, use_noise=use_noise, activation=activation, demodulate=demodulate,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper_Style(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper_Style(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x, style):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g), style)
        x_l, x_g = self.conv2((x_l, x_g), style)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,)
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,)
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,)

    def forward(self, x, img, style, skip=None, noise_mode='random'):
        x = self.conv0(x, style[:, 0], noise_mode=noise_mode)
        x = x + skip if skip is not None else x
        x = self.conv1(x, style[:, 1], noise_mode=noise_mode)
        img = self.toRGB(x, style[:, 2], skip=img)

        return x, img


class StyleConv(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        style_dim,                      # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        demodulate      = True,         # perform demodulation
    ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=demodulate,
                                    up=up,
                                    resample_filter=resample_filter,
                                    conv_clamp=conv_clamp)

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.conv_clamp = conv_clamp

    def forward(self, x, style, noise_mode='random', gain=1):
        x = self.conv(x, style)

        assert noise_mode in ['random', 'const', 'none']

        if self.use_noise:
            if noise_mode == 'random':
                xh, xw = x.size()[-2:]
                noise = torch.randn([x.shape[0], 1, xh, xw], device=x.device) \
                        * self.noise_strength
            if noise_mode == 'const':
                noise = self.noise_const * self.noise_strength
            x = x + noise

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act.bias_act(x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp)

        return out


class ToRGB(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 style_dim,
                 kernel_size=1,
                 resample_filter=[1,3,3,1],
                 conv_clamp=None,
                 demodulate=False):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=demodulate,
                                    resample_filter=resample_filter,
                                    conv_clamp=conv_clamp)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

    def forward(self, x, style, skip=None):
        x = self.conv(x, style)
        out = bias_act.bias_act(x, self.bias, clamp=self.conv_clamp)

        if skip is not None:
            if skip.shape != out.shape:
                skip = upfirdn2d.upsample2d(skip, self.resample_filter)
            out = out + skip

        return out
    

class ModulatedConv2d(nn.Module):
    def __init__(self,
                 in_channels,                   # Number of input channels.
                 out_channels,                  # Number of output channels.
                 kernel_size,                   # Width and height of the convolution kernel.
                 style_dim,                     # dimension of the style code
                 demodulate=True,               # perfrom demodulation
                 up=1,                          # Integer upsampling factor.
                 down=1,                        # Integer downsampling factor.
                 resample_filter=[1,3,3,1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,               # Clamp the output to +-X, None = disable clamping.
                 ):
        super().__init__()
        self.demodulate = demodulate

        self.weight = torch.nn.Parameter(torch.randn([1, out_channels, in_channels, kernel_size, kernel_size]))
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        self.affine = FullyConnectedLayer(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight * self.weight_gain * style

        if self.demodulate:
            decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, batch * in_channels, height, width)
        x = conv2d_resample.conv2d_resample(x=x, w=weight, f=self.resample_filter, up=self.up, down=self.down,
                                            padding=self.padding, groups=batch)
        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out


class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 in_features,                # Number of input features.
                 out_features,               # Number of output features.
                 bias            = True,     # Apply additive bias before the activation function?
                 activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier   = 1,        # Learning rate multiplier.
                 bias_init       = 0,        # Initial value for the additive bias.
                 ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.activation = activation

        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            # out = torch.addmm(b.unsqueeze(0), x, w.t())
            x = x.matmul(w.t())
            out = x + b.reshape([-1 if i == x.ndim-1 else 1 for i in range(x.ndim)])
        else:
            x = x.matmul(w.t())
            out = bias_act.bias_act(x, b, act=self.activation, dim=x.ndim-1)
        return out