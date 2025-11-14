# Shree KRISHNAya Namaha
# Differentiable warper implemented in PyTorch. Warping is done on batches.
# Tested on PyTorch 1.8.1
# Author: Nagabhushan S N
# Last Modified: 27/09/2021


# splatting_function:
# https://github.com/CompVis/geometry-free-view-synthesis/blob/master/geofree/modules/warp/midas.py


import datetime
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional

import numpy
import skimage.io
import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
import kornia.morphology as km

from splatting import splatting_function


class Warper:
    def __init__(self, resolution: tuple = None, eks=-1, gks=-1):
        self.resolution = resolution
        self.eks = eks
        self.gks = gks
        if self.eks > 0:
            print("Use Erode!")
        if self.gks > 0:
            print("Use Gaussian Blur!")

    def process_mask(self, img, vis_mask):
        if self.eks > 0:
            kernel = torch.ones(self.eks, self.eks).to(vis_mask.device)
            vis_mask = km.erosion(vis_mask, kernel)
        if self.gks > 0:
            vis_mask = transforms.GaussianBlur(kernel_size=self.gks)(vis_mask)
            vis_mask = torch.where(vis_mask > 0.5, torch.tensor(1.0).to(vis_mask), torch.tensor(0.0).to(vis_mask))
        masked_img = img * vis_mask
        mask = (masked_img == 0.).all(dim=1, keepdim=True).float().to(masked_img)

        return masked_img, 1 - mask

    def forward_warp(self, img1: torch.Tensor, depth1: torch.Tensor,
                     c1: torch.Tensor, c2: torch.Tensor,
                     in1: Optional[torch.Tensor]=None, in2: Optional[torch.Tensor]=None,
                     func="summation") -> \
                     Tuple[torch.Tensor, torch.Tensor]:

        if self.resolution is not None:
            assert img1.shape[2:4] == self.resolution
        b, c, h, w = img1.shape
        if depth1.shape[2:4] != (h, w):
            depth1 = F.interpolate(depth1, size=(h, w), mode='bicubic', align_corners=False)  # bicubic / bilinear
        if in1 is None:
            in1 = torch.tensor([[4.2647*w, 0, 0.5*w], [0, 4.2647*h, 0.5*h], [0, 0, 1]]).unsqueeze(0).repeat(b, 1, 1).to(img1)
        if in2 is None:
            in2 = in1.clone()
        
        ex1 = c1[:, :16].view(b, 4, 4)
        ex2 = c2[:, :16].view(b, 4, 4)

        # assert img1.shape == (b, 3, h, w)
        # assert depth1.shape == (b, 1, h, w)
        assert ex1.shape == (b, 4, 4)
        assert ex2.shape == (b, 4, 4)
        assert in1.shape == (b, 3, 3)
        assert in2.shape == (b, 3, 3)


        ## compute_transformed_points
        transformation1 = torch.linalg.inv(ex1)
        transformation2 = torch.linalg.inv(ex2)
        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, in1, in2)  # (b, h, w, 3, 1)
        trans_coordinates = trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, :, 2, 0].unsqueeze(1)  # (b, 1, h, w)
        trans_coordinates = trans_coordinates.permute(0, 3, 1, 2)  # (b, 2, h, w)
        grid = self.create_grid(b, h, w).to(trans_coordinates)

        flow12 = trans_coordinates - grid

        importance = 1.0 / trans_depth1
        importance_min = importance.amin((1,2,3),keepdim=True)
        importance_max = importance.amax((1,2,3),keepdim=True)
        importance=(importance-importance_min)/(importance_max-importance_min+1e-6)*10-10
        importance = importance.exp()


        if func == "summation":
            input_data = torch.cat([importance*img1, importance], 1)
            output_data = splatting_function("summation", input_data, flow12)

            num = output_data[:,:-1,:,:]
            nom = output_data[:,-1:,:,:]

            #rendered = num/(nom+1e-7)
            warped = num / nom.clamp(min=1e-8)
            mask = (warped == 0.).all(dim=1, keepdim=True).to(img1)

        elif func == "softmax":
            warped = splatting_function("softmax", img1, flow12, importance)
            mask = (warped == 0.).all(dim=1, keepdim=True).to(img1)

        else:
            raise NotImplementedError("Only support summation and softmax")
        
        vis_mask = 1 - mask
        
        if self.eks > 0 or self.gks > 0:
            warped, vis_mask = self.process_mask(warped, vis_mask)
        
        return warped, vis_mask, flow12


    def compute_transformed_points(self, depth1: torch.Tensor, transformation1: torch.Tensor, transformation2: torch.Tensor,
                                   intrinsic1: torch.Tensor, intrinsic2: Optional[torch.Tensor]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)

        x1d = torch.arange(0, w)[None]  # * (1./w) + (0.5/w)  ##
        y1d = torch.arange(0, h)[:, None]  # * (1./h) + (0.5/h)  ##
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (b, h, w, 1, 1)
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]  # * (1. / w) + (0.5 / w)  ##
        y_1d = torch.arange(0, h)[:, None]  # * (1. / h) + (0.5 / h)  ##
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/cuda:0
        :return:
        """
        if device == 'cpu':
            device = torch.device('cpu')
        # elif device.startswith('gpu') and torch.cuda.is_available():
        #     gpu_num = int(device[3:])
        #     device = torch.device(f'cuda:{gpu_num}')
        elif device.startswith('cuda') and torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device('cpu')
        return device