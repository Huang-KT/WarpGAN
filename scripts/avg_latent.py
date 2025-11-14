import numpy as np
import torch
import pickle
from PIL import Image

import sys
sys.path.append(".")
sys.path.append("..")

import dnnlib
from torch_utils import misc
import legacy
from models.eg3d.triplane import TriPlaneGenerator
from models.eg3d.camera_utils import LookAtPoseSampler
from configs.paths_config import model_paths
from utils import common

device = 'cuda:0'

cam_pivot = torch.tensor([0, 0, 0.2], device=device)
conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=2.7, device=device)
intrinsics = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).to(device)
c = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)


def get_avg_latent(G, w_avg_samples=10000):
    print(f'Computing avg_latent using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)

    conditioning_params = c.expand([w_avg_samples, c.shape[1]]).to(device)

    truncation_psi = 1
    truncation_cutoff = 14
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)  # [N, L, C]
    # w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    # w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    # w_avg_tensor = torch.from_numpy(w_avg).to(device)
    w_avg = torch.mean(w_samples[:, 0, :], dim=0, keepdim=True)  # [1, C]
    return w_avg

if __name__ == '__main__':
    ## 1. load the generator from the pickle file
    pkl_path = model_paths["eg3d_ffhq_lpff"]
    with dnnlib.util.open_url(pkl_path) as f:
        pickle_dict = legacy.load_network_pkl(f)
        G = pickle_dict["G_ema"].to(device)
        # self.latent_avg = pickle_dict['latent_avg'].to(device).repeat(self.opts.n_styles, 1)

    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    decoder = G_new
    print("Loaded generator from", pkl_path)

    ## 2. load the generator from the pickle file
    # ckpt = torch.load(model_paths["eg3d_ffhq_pth"])
    # # self.latent_avg = ckpt['latent_avg'].to(self.opts.device).repeat(self.opts.n_styles, 1)

    # init_args = ()
    # init_kwargs = ckpt['init_kwargs']
    # decoder = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(device)
    # decoder.neural_rendering_resolution = 128
    # decoder.load_state_dict(ckpt['G_ema'], strict=False)
    # decoder.requires_grad_(False)


    avg_latent = get_avg_latent(decoder)
    save_path = '../workspace/pretrained_models/eg3d/ffhq_lpff/latent_avg_lpff.pt'
    torch.save(avg_latent, save_path)
    print('Save avg_latent to', save_path)

    # img = decoder.synthesis(avg_latent.unsqueeze(1).repeat(1, 14, 1), c.to(device), noise_mode='const')['image'][0]
    # result = common.tensor2im(img)
    # Image.fromarray(np.array(result)).save('../workspace/experiments/avg_img/avg_img_ffhq_rebalanced.png')
    # print('avg image saved!')