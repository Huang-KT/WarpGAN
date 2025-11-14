# input: none
# output: src, src_depth_gt, src_c
#         codes, src_hat, src_depth
#         target, target_depth_gt, target_c
#         target_hat, target_depth

import os
import json
import random
import numpy as np
import cv2
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from PIL import Image
import pytz
import datetime
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

import sys
sys.path.append(".")
sys.path.append("..")

import pickle
import dnnlib
from torch_utils import misc
import legacy

from configs.paths_config import dataset_paths
from configs.paths_config import model_paths

from models.wplusnet import WplusNet
from models.eg3d.triplane import TriPlaneGenerator
from models.eg3d.camera_utils import LookAtPoseSampler

from utils import common


SEED = 2107
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

EPS = 1


def tensor2im(var):
    var = var.cpu().detach()
    var = var.permute(1, 2, 0).numpy()
    var = (var + 1) / 2
    var = np.clip(var, 0, 1)
    var = (var * 255).astype(np.uint8)
    return var

class SynthData:
    def __init__(self, opts):
        self.opts = opts

        data_size = 100000
        self.opts.exp_dir = './data/SynthData' + str(data_size) + '_rebalanced'
        self.opts.data.synth.batch_size = 4
        self.max_iter = data_size // self.opts.data.synth.batch_size

        self.device = self.opts.device

        # Initialize network
        self.gan = WplusNet(self.opts.gan).to(self.device)
        self.gan.eval()

        self.eg3d_type = "plus_r"  # ori_r / plus_r
        print("eg3d_type:", self.eg3d_type)
        self.load_decoder()

        if 'plus' in self.eg3d_type:
            self.cam_json = 'dataset_plus_rebalanced.json'
        else:
            self.cam_json = 'dataset_rebalanced.json'
        print("cam_json:", self.cam_json)
        self.ffhq_cam_list = self.get_ffhq_camera_params()

        self.gan.psp_encoder.requires_grad_(False)

        if os.path.exists(self.opts.exp_dir):
            raise Exception('Oops... {} already exists'.format(opts.exp_dir))
        os.makedirs(opts.exp_dir, exist_ok=False)


    def forward(self):

        current_iter = 0

        for _ in tqdm(range(self.max_iter), total=self.max_iter, desc="Synthesizing data"):
            with torch.no_grad():
                synth_batch = self.synth_data()

            src_img, target_img = synth_batch['src_img'], synth_batch['target_img']
            src_img_256, target_img_256 = synth_batch['src_img_256'], synth_batch['target_img_256']
            src_c, target_c = synth_batch['src_c'], synth_batch['target_c']
            src_depth, target_depth = synth_batch['src_depth'], synth_batch['target_depth']


            with torch.no_grad():
                outs = self.gan(src_img_256, src_c, novel_view_camera_params=target_c)

            codes = outs["codes"]
            y_hat, depth = outs["y_hat"], outs["depth"]
            y_hat_novel, depth_novel = outs["y_hat_novel"], outs["depth_novel"]

            ## Save results
            for i in range(src_img.size(0)):
                save_dir = os.path.join(self.opts.exp_dir, f'{current_iter:06d}')
                os.makedirs(save_dir, exist_ok=True)
                
                save_path_img = [os.path.join(save_dir, 'src.png'), os.path.join(save_dir, 'src_hat.png'), os.path.join(save_dir, 'target.png'), os.path.join(save_dir, 'target_hat.png')]
                save_img = [src_img[i], y_hat[i], target_img[i], y_hat_novel[i]]

                for path, img in zip(save_path_img, save_img):
                    # common.tensor2im(img).save(path)
                    img = tensor2im(img)
                    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                save_path_code = os.path.join(save_dir, 'codes.pt')
                torch.save(codes[i], save_path_code)
                
                save_path_depth = [os.path.join(save_dir, 'src_depth_gt.pt'), os.path.join(save_dir, 'src_depth.pt'), os.path.join(save_dir, 'target_depth_gt.pt'), os.path.join(save_dir, 'target_depth.pt')]
                save_depth = [src_depth[i], depth[i], target_depth[i], depth_novel[i]]
                for path, pt in zip(save_path_depth, save_depth):
                    torch.save(pt, path)
                
                save_path_c = [os.path.join(save_dir, 'src_c.pt'), os.path.join(save_dir, 'target_c.pt')]
                save_c = [src_c[i], target_c[i]]
                for path, pt in zip(save_path_c, save_c):
                    torch.save(pt, path)
                
                current_iter += 1
        
        print("Data synthesis done!")

    def sample_camera_poses(self, N):
        sampled_poses = random.sample(self.ffhq_cam_list, N)
        sampled_poses = torch.from_numpy(np.array(sampled_poses).reshape(N, -1)).to(self.device).float()
        return sampled_poses

    def get_pose(self, cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7):
        
        if yaw is None:
            yaw = np.random.uniform(-yaw_range, yaw_range)
        if pitch is None:
            pitch = np.random.uniform(-pitch_range, pitch_range)

        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=self.device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
        return c

    def get_ffhq_camera_params(self):
        with open(os.path.join(dataset_paths['train'], self.cam_json), 'r') as f:
            ffhq_cam_list = json.load(f)['labels']
        
        ffhq_cam_list = [[x[1]] for x in ffhq_cam_list]
        return ffhq_cam_list

    def load_decoder(self):
        if self.eg3d_type == 'ori_r':
            pkl_path = model_paths["eg3d_ffhq_rebalanced"]
        elif self.eg3d_type == 'plus_r':
            pkl_path = model_paths["eg3d_ffhq_lpff_rebalanced"]
        elif self.eg3d_type == 'cat':
            pkl_path = model_paths["eg3d_afhqcats"]
        else:
            raise ValueError(f"Unknown eg3d_type: {self.eg3d_type}")
        
        fixed_planes = True if 'plus' in self.eg3d_type else False
        
        with dnnlib.util.open_url(pkl_path) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(self.opts.device)
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs, fixed_planes=fixed_planes).eval().requires_grad_(False).to(self.opts.device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        self.decoder = G_new
        print("Loading eg3d generator from", pkl_path)

    def synth_data(self):

        b = self.opts.data.synth.batch_size
        per_num = 2

        conditioning_c = self.get_pose(cam_pivot=torch.tensor([0, 0, 0.2]).to(self.device),
                                       intrinsics=torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).to(self.device),
                                       yaw=0, pitch=0).repeat(b, 1)  # b, 25

        trunc_psi, trunc_cutoff = 0.7, 14
        with torch.no_grad():
            if self.opts.data.synth.zplus:
                zs = torch.randn((b, 14, self.decoder.z_dim)).to(self.device)
                ws = []
                for i in range(14):
                    w = self.decoder.mapping(
                        zs[:, i, :], conditioning_c,
                        truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff,
                    )
                    ws.append(w[:, 0, :])
                ws = torch.stack(ws, dim=1)
            else:
                z = torch.randn((b, self.decoder.z_dim)).to(self.device)
                ws = self.decoder.mapping(
                    z, conditioning_c,
                    truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff,
                )

        ## Method 1: get_pose()
        synth_c = []
        for i in range(b * per_num):
            c = self.get_pose(cam_pivot=torch.tensor([0, 0, 0.2]).to(self.device),
                              intrinsics=torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).to(self.device),
                              yaw_range=0.5, pitch_range=0.5)
            synth_c.append(c)
        synth_c = torch.cat(synth_c, dim=0)  # b * per_num, 25

        ## Method 2: sample_camera_poses()
        # synth_c = self.sample_camera_poses(b * per_num)  # b * per_num, 25


        ws = ws.unsqueeze(1).repeat(1, per_num, 1, 1).view(b * per_num, 14, self.decoder.z_dim)  # b, 14, 512 -> b * per_num, 14, 512

        with torch.no_grad():
            synth_outs = self.decoder.synthesis(ws, synth_c, noise_mode='const')

        # img
        synth_imgs = synth_outs['image']  # b * per_num, 3, 256, 256
        _, _c, _h, _w = synth_imgs.shape  # b * per_num, 3, h, w
        synth_imgs = synth_imgs.view(b, per_num, _c, _h, _w)  # b, per_num, 3, h, w
        # c
        synth_c = synth_c.view(b, per_num, -1)  # b, per_num, 25
        # depth
        synth_depth = synth_outs['image_depth']  # b * per_num, 1, 128, 128
        _, _c, _h, _w = synth_depth.shape
        synth_depth = synth_depth.view(b, per_num, _c, _h, _w)  # b, per_num, 1, 128, 128

        # Get source and target
        src_img, target_img = synth_imgs[:, 0, :, :, :], synth_imgs[:, 1, :, :, :]  # b, 3, h, w
        src_c, target_c = synth_c[:, 0, :], synth_c[:, 1, :]  # b, 25
        src_depth, target_depth = synth_depth[:, 0, :, :, :], synth_depth[:, 1, :, :, :]  # b, 1, 128, 128

        src_img_256 = F.adaptive_avg_pool2d(src_img, (256, 256))
        target_img_256 = F.adaptive_avg_pool2d(target_img, (256, 256))

        return {
            'src_img': src_img, 'target_img': target_img,
            'src_img_256':src_img_256, 'target_img_256': target_img_256,
            'src_c': src_c, 'target_c': target_c,
            'src_depth': src_depth, 'target_depth': target_depth
        }


@hydra.main(config_path="../configs", config_name="train_inpainting")
def main(opts: DictConfig):

    SynthData(opts).forward()


if __name__ == '__main__':
	main()
