import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.goae.goae import GOAEncoder
from configs.swin_config import get_config
from models.eg3d.triplane import TriPlaneGenerator
from configs.paths_config import model_paths
import pickle
import dnnlib
from torch_utils import misc
import legacy


class WplusNet(nn.Module):
    def __init__(self, opts):
        super(WplusNet, self).__init__()
        self.opts = opts
        self.set_psp_encoder()
        self.set_eg3d_generator()

        self.load_weights()

    def forward(self, x, camera_params, novel_view_camera_params=None):

        outputs = self.get_initial_inversion(x, camera_params.clone().detach(), novel_view_camera_params)
        
        return outputs
    
    def encoder_forward(self, x):
        codes = self.psp_encoder(x)
        # add to average latent code
        codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1).to(codes)
        return codes

    def get_initial_inversion(self, x, camera_params, novel_view_camera_params):
        codes = self.encoder_forward(x)
        outs = self.decoder.synthesis(codes, camera_params, noise_mode='const')
        y_hat = outs['image']
        y_hat_resized = F.adaptive_avg_pool2d(y_hat, (256, 256))
        outputs = {
            "codes": codes,
            "y_hat": y_hat, "y_hat_resized": y_hat_resized,
            "depth": outs["image_depth"],
        }
        if novel_view_camera_params is not None:
            if not isinstance(novel_view_camera_params, list):
                novel_view_camera_params = [novel_view_camera_params]

            y_hat_novel_list, y_hat_novel_resized_list, depth_novel_list = [], [], []
            for novel_c in novel_view_camera_params:
                outs_novel = self.decoder.synthesis(codes, novel_c, noise_mode='const')
                y_hat_novel = outs_novel['image']
                y_hat_novel_resized =  F.adaptive_avg_pool2d(y_hat_novel, (256, 256))

                y_hat_novel_list.append(y_hat_novel)
                y_hat_novel_resized_list.append(y_hat_novel_resized)
                depth_novel_list.append(outs_novel["image_depth"])

            if len(y_hat_novel_list) == 1:
                y_hat_novel_list, y_hat_novel_resized_list, depth_novel_list = y_hat_novel_list[0], y_hat_novel_resized_list[0], depth_novel_list[0]
            outputs["y_hat_novel"] = y_hat_novel_list
            outputs["y_hat_novel_resized"] = y_hat_novel_resized_list
            outputs["depth_novel"] = depth_novel_list

        return outputs

    def set_psp_encoder(self):
        print("Encoder type: ", self.opts.encoder_type)
        if self.opts.encoder_type == 'psp':
            self.psp_encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts.psp)
        elif self.opts.encoder_type == 'goae':
            swin_config = get_config(self.opts.goae)
            self.psp_encoder = GOAEncoder(swin_config, mlp_layer=self.opts.goae.mlp_layer, stage_list=[10000, 20000, 30000])

    def set_eg3d_generator(self):
        ## 1、load the generator from the .pth file
        if self.opts.eg3d_type == 'ori_pth':
            ckpt = torch.load(model_paths["eg3d_ffhq_pth"], map_location='cpu')
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device).repeat(self.opts.psp.n_styles, 1)
            init_args = ()
            init_kwargs = ckpt['init_kwargs']
            self.decoder = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(self.opts.device)
            self.decoder.neural_rendering_resolution = 128
            self.decoder.load_state_dict(ckpt['G_ema'], strict=False)
            self.decoder.requires_grad_(False)
            print("Loading eg3d generator from", model_paths["eg3d_ffhq_pth"])
        else:
        ## 2. load the generator from the pickle file
            if self.opts.eg3d_type == 'ori':
                pkl_path, latent_avg_path = model_paths["eg3d_ffhq"], model_paths['latent_avg']
            elif self.opts.eg3d_type == 'ori_r':
                pkl_path, latent_avg_path = model_paths["eg3d_ffhq_rebalanced"], model_paths['latent_avg_rebalanced']
            elif self.opts.eg3d_type == 'plus':
                pkl_path, latent_avg_path = model_paths["eg3d_ffhq_lpff"], model_paths['latent_avg_plus']
            elif self.opts.eg3d_type == 'plus_r':
                pkl_path, latent_avg_path = model_paths["eg3d_ffhq_lpff_rebalanced"], model_paths['latent_avg_plus_rebalanced']
            elif self.opts.eg3d_type == 'cat':
                pkl_path, latent_avg_path = model_paths["eg3d_afhqcats"], None
            else:
                raise ValueError(f"Unknown eg3d_type: {self.opts.eg3d_type}")
            
            fixed_planes = True if 'plus' in self.opts.eg3d_type else False
            
            with dnnlib.util.open_url(pkl_path) as f:
                G = legacy.load_network_pkl(f)["G_ema"].to(self.opts.device)
            G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs, fixed_planes=fixed_planes).eval().requires_grad_(False).to(self.opts.device)
            misc.copy_params_and_buffers(G, G_new, require_all=True)
            G_new.neural_rendering_resolution = G.neural_rendering_resolution
            G_new.rendering_kwargs = G.rendering_kwargs
            self.decoder = G_new
            print("Loading eg3d generator from", pkl_path)
            
            self.latent_avg = torch.load(latent_avg_path, map_location='cpu')  # .to(self.opts.device)  # .repeat(self.opts.psp.n_styles, 1)
            print("Loading avg latent from", latent_avg_path)
            # self.latent_avg = self.decoder.backbone.mapping.w_avg
            # print("Loading avg latent from self.decoder.backbone.mapping.w_avg")

    def load_weights(self):

        if self.opts.checkpoint_path is None:
            if self.opts.encoder_type == 'psp':
                print('Loading encoders weights from irse50!')
                encoder_ckpt = torch.load(model_paths['ir_se50'], map_location='cpu')
                self.psp_encoder.load_state_dict(encoder_ckpt, strict=False)
            elif self.opts.encoder_type == 'goae':
                print('Loading encoders weights from goae!')
                encoder_ckpt = torch.load(model_paths['goae'], map_location='cpu')
                self.psp_encoder.load_state_dict(encoder_ckpt, strict=True)
        else:
            checkpoint = torch.load(self.opts.checkpoint_path, map_location='cpu')[self.opts.state_dict_key]
            self.psp_encoder.load_state_dict(checkpoint, strict=True)  ##
            print(f"Loading gan weights from {self.opts.checkpoint_path}")