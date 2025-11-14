# input: x, c
# output: x, c, codes, y_hat, depth;
#         c1, y_hat_novel1, depth_novel1;
#         c2, y_hat_novel2, depth_novel2;
#         ... ...

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

from datasets.dataset_inpainting_mirror import ImageFolderDataset
from configs.paths_config import dataset_paths
from configs.paths_config import model_paths

from models.wplusnet import WplusNet
from models.eg3d.camera_utils import LookAtPoseSampler

from utils import common, train_utils


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

        self.dataset_path = './data/celeba-hq_1000_align'
        self.opts.exp_dir = './data/celeba-hq_1000_static_rebalanced'
        
        # self.dataset_path = './data/FFHQ-LPFF-EG3D_all'
        # self.opts.exp_dir = './data/FFHQ-LPFF-EG3D_all_static'

        # self.dataset_path = './data/FFHQ-EG3D_all'
        # self.opts.exp_dir = './data/FFHQ-EG3D_all_static_rebalanced'

        self.opts.data.batch_size = 4
        self.opts.data.workers = 4

        self.device = self.opts.device

        # Initialize network
        self.gan = WplusNet(self.opts.gan).to(self.device)
        self.gan.eval()

        self.cam_json = 'dataset_rebalanced.json'
        self.ffhq_cam_list = self.get_ffhq_camera_params()

        self.gan.psp_encoder.requires_grad_(False)

        # Initialize dataset
        self.dataset_json = 'dataset.json'
        self.configure_datasets()
        self.configure_dataloaders()

        if os.path.exists(self.opts.exp_dir):
            raise Exception('Oops... {} already exists'.format(opts.exp_dir))
        os.makedirs(opts.exp_dir, exist_ok=False)


    def forward(self):

        for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Novel View"):

            x_256, x, c, x_mirror_256, x_mirror, c_mirror, conf_map_mirror, fname = batch
            x_256, x, c = x_256.to(self.device).float(), x.to(self.device).float(), c.to(self.device).float()
            x_mirror_256, x_mirror, c_mirror = x_mirror_256.to(self.device).float(), x_mirror.to(self.device).float(), c_mirror.to(self.device).float()

            novel_view_num = 3
            c_novel_list = [c_mirror]
            for _ in range(novel_view_num):
                c_novel = self.sample_camera_poses(x.size(0))
                c_novel_list.append(c_novel)
            with torch.no_grad():
                outs = self.gan(x_256, c, novel_view_camera_params=c_novel_list)

            codes = outs["codes"]
            y_hat, depth = outs["y_hat"], outs["depth"]
            y_hat_novel_list, depth_novel_list = outs["y_hat_novel"], outs["depth_novel"]

            ## Save results
            for i in range(x.size(0)):
                save_dir = os.path.join(self.opts.exp_dir, fname[i])
                os.makedirs(save_dir, exist_ok=True)
                
                save_path_img = [os.path.join(save_dir, 'x.png'), os.path.join(save_dir, 'y_hat.png')]
                save_img = [x[i], y_hat[i]]
                for view in range(novel_view_num+1):
                    if view == 0:
                        save_path_img.append(os.path.join(save_dir, f'y_hat_mirror.png'))
                    else:
                        save_path_img.append(os.path.join(save_dir, f'y_hat_novel_{view}.png'))
                    save_img.append(y_hat_novel_list[view][i])
                for path, img in zip(save_path_img, save_img):
                    # common.tensor2im(img).save(path)
                    img = tensor2im(img)
                    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                save_path_code = os.path.join(save_dir, 'codes.pt')
                torch.save(codes[i], save_path_code)
                
                save_path_depth, save_path_c = [os.path.join(save_dir, 'depth.pt')], [os.path.join(save_dir, 'c.pt')]
                save_depth, save_c = [depth[i]], [c[i]]
                for view in range(novel_view_num+1):
                    if view == 0:
                        save_path_depth.append(os.path.join(save_dir, 'depth_mirror.pt'))
                        save_path_c.append(os.path.join(save_dir, 'c_mirror.pt'))
                    else:
                        save_path_depth.append(os.path.join(save_dir, f'depth_novel_{view}.pt'))
                        save_path_c.append(os.path.join(save_dir, f'c_novel_{view}.pt'))
                    save_depth.append(depth_novel_list[view][i])  ####
                    save_c.append(c_novel_list[view][i])  ####
                for path, pt in zip(save_path_depth, save_depth):
                    torch.save(pt, path)
                for path, pt in zip(save_path_c, save_c):
                    torch.save(pt, path)
        
        print("Generate novel views done!")


    def configure_datasets(self):

        self.dataset = ImageFolderDataset(path=self.dataset_path,
                                          resolution=None,
                                          use_labels=True,
                                          load_conf_map=False,  # True
                                          datast_json=self.dataset_json)
        
        print(f"Number of training samples: {len(self.dataset)}")

    def configure_dataloaders(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=int(self.opts.data.batch_size),
                                     shuffle=False,
                                     num_workers=int(self.opts.data.workers),
                                     drop_last=False)

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


@hydra.main(config_path="../configs", config_name="train_inpainting")
def main(opts: DictConfig):

    SynthData(opts).forward()


if __name__ == '__main__':
	main()
