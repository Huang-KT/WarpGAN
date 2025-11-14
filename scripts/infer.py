import os
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import time
from tqdm import tqdm
import imageio
import json
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import kornia.morphology as km

import sys
sys.path.append(".")
sys.path.append("..")

# from datasets.dataset_inpainting_mirror import ImageFolderDataset
from datasets.dataset_vanilla import ImageFolderDataset
from configs.paths_config import dataset_paths
from models.wplusnet import WplusNet
from models.eg3d.camera_utils import LookAtPoseSampler
from models.saicinpainting.training.modules import make_generator
from utils.warp.Splatting import Warper
from models.saicinpainting.utils import set_requires_grad
from utils.common import add_grid_lines

from utils.depth_utils import calibrate_disparity


SEED = 2107
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


class Inference:
    def __init__(self, opts):
        self.opts = opts
        self.device = self.opts.device

        self.gan = WplusNet(self.opts.gan).to(self.device)
        self.gan.eval()
        set_requires_grad(self.gan, False)

        if self.opts.infer_type == 'inpaint':
            self.inpaintor = make_generator(self.opts, **self.opts.generator).to(self.device)
            inpaintor_state = torch.load(self.opts.ckpt_inpaintor, map_location='cpu')['inpaintor_state_dict']
            self.inpaintor.load_state_dict(inpaintor_state, strict=True)
            print(f"Loading inpaintor weights from {self.opts.ckpt_inpaintor}")
            self.inpaintor.eval()
            set_requires_grad(self.inpaintor, False)

        if self.opts.infer_type == 'inpaint' or self.opts.infer_type == 'coarse' or self.opts.infer_type == 'warp':
            self.warper = Warper()
        
        if self.opts.novel_view == "video":
            cam_pivot = torch.Tensor([0., 0., 0.2]).to(self.device)
            pitch_range, yaw_range = 0.25, 0.35
            intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=self.device)
            num_keyframes = 30
            self.render_poses = []
            for frame_idx in range(num_keyframes):
                cam2world_pose = LookAtPoseSampler.sample(3.14/2        + yaw_range   * np.sin(2 * 3.14 * frame_idx / (num_keyframes)),
                                                            3.14/2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes)),
                                                            cam_pivot, radius=2.7, device=self.device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                self.render_poses.append(c)

        
        print("Inference Type:", self.opts.infer_type)

        self.dataset_json = 'dataset.json'
        self.cam_json = self.opts.data.cam_json
        # self.ffhq_cam_list = self.get_ffhq_camera_params()

        self.dataset = ImageFolderDataset(path=self.opts.data.path,
                                          use_mirror=True,
                                          use_labels=True,
                                          load_depth=self.opts.use_depth_est,
                                          datast_json=self.dataset_json,
                                          max_size=1000,
                                          get_c_novel=False,)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.opts.data.batch_size, shuffle=True,
                                     num_workers=self.opts.data.num_workers, drop_last=False)
        
        self.use_depth_est = self.opts.use_depth_est
        print(f"Using depth estimation: {self.use_depth_est}")

        # if self.opts.save_result:
        # if os.path.exists(self.opts.exp_dir):
        #     raise Exception('Oops... {} already exists'.format(self.opts.exp_dir))
        # os.makedirs(self.opts.exp_dir, exist_ok=False)
        print("Exp dir:", self.opts.exp_dir)

    def get_inp(self, img, inversion, mask, cat_inv=True):
        masked_img, mask = self.process_mask(img, mask)
        if self.opts.warp.hybrid:
            masked_img = img * (1 - mask) + inversion * mask
            
        if cat_inv:
            inp = torch.cat([masked_img, inversion, mask], dim=1)
        else:
            inp = torch.cat([masked_img, mask], dim=1)
        return inp, masked_img, mask

    def process_mask(self, img, mask):
        eks = self.opts.warp.erode_kernel
        gks = self.opts.warp.gaussian_blur_kernel

        if eks > 0 or gks > 0:
            vis_mask = 1 - mask.clone()
            if eks > 0:
                kernel = torch.ones(eks, eks).to(vis_mask.device)
                vis_mask = km.erosion(vis_mask, kernel)
            if gks > 0:
                vis_mask = transforms.GaussianBlur(kernel_size=gks)(vis_mask)
                # vis_mask = torch.where(vis_mask > 0.5, torch.tensor(1.0).to(vis_mask), torch.tensor(0.0).to(vis_mask))
            mask = 1 - vis_mask
            masked_img = img * (1 - mask)
            # mask = (masked_img == 0.).all(dim=1, keepdim=True).float().to(masked_img)

        else:
            masked_img = img * (1 - mask)

        return masked_img, mask
    
    def inpaint(self, batch_inp):
        img, inversion, mask = batch_inp['image'], batch_inp['inversion'], batch_inp['mask']
        ws = batch_inp['ws']

        img = ((img + 1) / 2).clamp(0, 1)
        inversion = ((inversion + 1) / 2).clamp(0, 1)

        inp, masked_img, mask = self.get_inp(img, inversion, mask, self.opts.generator.cat_inv)

        if self.opts.infer_type == 'coarse':
            return masked_img * 2 - 1

        if self.opts.generator.input_mirror == 'cat' or self.opts.generator.input_mirror == 'condition':
            img_mirror, mask_mirror = batch_inp['image_mirror'], batch_inp['mask_mirror']
            img_mirror = ((img_mirror + 1) / 2).clamp(0, 1)
            inp_mirror, masked_img_mirror, mask_mirror = self.get_inp(img_mirror, inversion, mask_mirror, cat_inv=self.opts.generator.cat_inv and self.opts.generator.input_mirror=='condition')
            inp = torch.cat([inp, inp_mirror], dim=1)
        else:
            masked_img_mirror, mask_mirror = None, None

        if self.opts.generator.kind == 'ffc_resnet':
            pred = self.inpaintor(inp)
        elif self.opts.generator.kind == 'ffc_style_resnet':
            pred = self.inpaintor(inp, ws)
        else:
            raise NotImplementedError(f"Generator kind {self.opts.generator.kind} not implemented")

        return pred * 2 - 1, mask * 2 - 1

    def forward(self):

        for batch_idx, batch in enumerate(tqdm(self.dataloader)):

            if batch_idx < self.opts.data.skip_batch:
                continue

            x_256, x, c, x_mirror_256, x_mirror, c_mirror, conf_map_mirror, depth_est, c_novel, frame = batch
            x_256, x, c = x_256.to(self.device).float(), x.to(self.device).float(), c.to(self.device).float()
            x_mirror_256, x_mirror, c_mirror = x_mirror_256.to(self.device).float(), x_mirror.to(self.device).float(), c_mirror.to(self.device).float()
            c_novel = c_novel.to(self.device).float()

            if self.use_depth_est:
                depth_est = depth_est.to(self.device).float().unsqueeze(1)
                # depth_est = 1 / (depth_est + 1e-8)
                depth_est_mirror = torch.flip(depth_est, dims=[3])
            else:
                depth_est, depth_est_mirror = None, None

            if self.opts.novel_view == "random":
                c_novel = self.sample_camera_poses(x.size(0))

                res, res_mask, res_warp, _ = self.forward_single(x_256, x, c, x_mirror_256, x_mirror, c_mirror, conf_map_mirror, depth_est, depth_est_mirror, frame, c_novel)

                if self.opts.infer_type == 'warp':
                    res_warp = F.interpolate(res_warp, size=(256, 256), mode='bilinear', align_corners=False)
                    res_mask = F.interpolate(res_mask, size=(256, 256), mode='bilinear', align_corners=False)
                    
                    for i in range(res_warp.size(0)):
                        save_warp_dir = os.path.join(self.opts.exp_dir, 'img')
                        os.makedirs(save_warp_dir, exist_ok=True)
                        save_warp_path = os.path.join(save_warp_dir, f'{frame[i]}.png')
                        cv2.imwrite(save_warp_path, cv2.cvtColor(self.tensor2im(res_warp[i]), cv2.COLOR_RGB2BGR))
                        
                        save_mask_dir = os.path.join(self.opts.exp_dir, 'mask')
                        os.makedirs(save_mask_dir, exist_ok=True)
                        save_mask_path = os.path.join(save_mask_dir, f'{frame[i]}_mask.png')
                        # cv2.imwrite(save_mask_path, cv2.cvtColor(self.tensor2im01(res_mask[i].repeat(3, 1, 1)), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(save_mask_path, self.tensor2im01(res_mask[i]))
                else:
                    # if self.opts.save_result:
                    for i in range(res.size(0)):
                        save_inp_dir = os.path.join(self.opts.exp_dir)
                        os.makedirs(save_inp_dir, exist_ok=True)
                        save_inp_path = os.path.join(save_inp_dir, f'{frame[i]}.png')
                        cv2.imwrite(save_inp_path, cv2.cvtColor(self.tensor2im(res[i]), cv2.COLOR_RGB2BGR))
                    
            elif self.opts.novel_view == "yaw" or self.opts.novel_view == "circle":
                
                if self.opts.novel_view == "yaw":

                    # inpaintor_yaw
                    # yaw_name_list = ["-30", "0", "30"]
                    # yaw_list = [-np.pi/6, 0, np.pi/6]
                    
                    res_list = [x]
                    res_mask_list = []
                    res_warp_img_list = [x]
                    for yaw_name, yaw in zip(yaw_name_list, yaw_list):
                        c_novel = self.get_pose(cam_pivot=torch.tensor([0, 0, 0.2]).to(self.device),
                                                intrinsics=torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).to(self.device),
                                                yaw=yaw, pitch=0).repeat(x.shape[0], 1)  # b, 25
                        # frame_yaw = [f"{f}_{yaw_name}" for f in frame]
                        res, res_mask, res_warp_img, depth = self.forward_single(x_256, x, c, x_mirror_256, x_mirror, c_mirror, conf_map_mirror, depth_est, depth_est_mirror, frame, c_novel)

                        for i in range(res.size(0)):
                            save_inp_dir = os.path.join(self.opts.exp_dir, 'single')
                            os.makedirs(save_inp_dir, exist_ok=True)

                            save_inp_path = os.path.join(save_inp_dir, f'{frame[i]}_{yaw_name}.png')
                            cv2.imwrite(save_inp_path, cv2.cvtColor(self.tensor2im(res[i]), cv2.COLOR_RGB2BGR))
                            
                            if self.opts.infer_type != 'inv':
                                save_mask_path = os.path.join(save_inp_dir, f'{frame[i]}_{yaw_name}_mask.png')
                                cv2.imwrite(save_mask_path, cv2.cvtColor(self.tensor2im(res_mask[i]), cv2.COLOR_RGB2BGR))
                                
                                save_warp_img_path = os.path.join(save_inp_dir, f'{frame[i]}_{yaw_name}_warp_img.png')
                                cv2.imwrite(save_warp_img_path, cv2.cvtColor(self.tensor2im(res_warp_img[i]), cv2.COLOR_RGB2BGR))

                        res_list.append(res)
                        if self.opts.infer_type != 'inv':
                            if len(res_mask_list) == 0:
                                res_mask_list.append(torch.zeros_like(res_mask).repeat(1, 3, 1, 1))
                            res_mask_list.append(res_mask.repeat(1, 3, 1, 1))
                            res_warp_img_list.append(res_warp_img)

                else:
                    angle_y, angle_p = 10, 10
                    y_p_list = [(0, 0), (-np.pi/angle_y, 0), (np.pi/angle_y, 0), (0, -np.pi/angle_p), (0, np.pi/angle_p)]
                    res_list = [x]
                    res_mask_list = []
                    res_warp_img_list = [x]
                    for y_p_i, (y, p) in enumerate(y_p_list):
                        c_novel = self.get_pose(cam_pivot=torch.tensor([0, 0, 0.2]).to(self.device),
                                                intrinsics=torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).to(self.device),
                                                yaw=y, pitch=p).repeat(x.shape[0], 1)  # b, 25
                        # frame_yaw = [f"{f}_{yaw_name}" for f in frame]
                        res, res_mask, res_warp_img, depth = self.forward_single(x_256, x, c, x_mirror_256, x_mirror, c_mirror, conf_map_mirror, depth_est, depth_est_mirror, frame, c_novel)

                        for i in range(res.size(0)):
                            save_inp_dir = os.path.join(self.opts.exp_dir, 'single')
                            os.makedirs(save_inp_dir, exist_ok=True)

                            save_inp_path = os.path.join(save_inp_dir, f'{frame[i]}_{y_p_i}.png')
                            cv2.imwrite(save_inp_path, cv2.cvtColor(self.tensor2im(res[i]), cv2.COLOR_RGB2BGR))
                            
                            if self.opts.infer_type != 'inv':
                                save_mask_path = os.path.join(save_inp_dir, f'{frame[i]}_{y_p_i}_mask.png')
                                cv2.imwrite(save_mask_path, cv2.cvtColor(self.tensor2im(res_mask[i]), cv2.COLOR_RGB2BGR))
                                
                                save_warp_img_path = os.path.join(save_inp_dir, f'{frame[i]}_{y_p_i}_warp_img.png')
                                cv2.imwrite(save_warp_img_path, cv2.cvtColor(self.tensor2im(res_warp_img[i]), cv2.COLOR_RGB2BGR))

                        res_list.append(res)
                        if self.opts.infer_type != 'inv':
                            if len(res_mask_list) == 0:
                                res_mask_list.append(torch.zeros_like(res_mask).repeat(1, 3, 1, 1))
                            res_mask_list.append(res_mask.repeat(1, 3, 1, 1))
                            res_warp_img_list.append(res_warp_img)
                
                res_cat = torch.cat(res_list, dim=3)
                if self.opts.infer_type != 'inv':
                    res_mask_cat = torch.cat(res_mask_list, dim=3)
                    res_warp_img_cat = torch.cat(res_warp_img_list, dim=3)
                    res_cat = torch.cat([res_cat, res_mask_cat, res_warp_img_cat], dim=2)
                for i in range(res_cat.size(0)):
                    save_inp_dir = os.path.join(self.opts.exp_dir, 'cat')
                    os.makedirs(save_inp_dir, exist_ok=True)
                    save_inp_path = os.path.join(save_inp_dir, f'{frame[i]}.png')
                    cv2.imwrite(save_inp_path, cv2.cvtColor(self.tensor2im(add_grid_lines(res_cat[i], 512)), cv2.COLOR_RGB2BGR))

                    save_ori_dir = os.path.join(self.opts.exp_dir, 'single')
                    os.makedirs(save_ori_dir, exist_ok=True)
                    save_ori_path = os.path.join(save_ori_dir, f'{frame[i]}.png')
                    cv2.imwrite(save_ori_path, cv2.cvtColor(self.tensor2im(x[i]), cv2.COLOR_RGB2BGR))

                    save_depth_path = os.path.join(save_ori_dir, f'{frame[i]}_depth.png')
                    cv2.imwrite(save_depth_path, cv2.cvtColor(self.tensor2im(depth[i].repeat(3, 1, 1), norm=True), cv2.COLOR_RGB2BGR))

            elif self.opts.novel_view == "video":

                warp_imgs, masks, inps = [], [], []
                for novel_pose in self.render_poses:
                    
                    res, res_mask, res_warp, _ = self.forward_single(x_256, x, c, x_mirror_256, x_mirror, c_mirror, conf_map_mirror, depth_est, depth_est_mirror, frame, novel_pose.repeat(x.size(0), 1))

                    warp_imgs.append(res_warp.detach())
                    masks.append(res_mask.detach() * 2 - 1)
                    inps.append(res.detach())


                for i in range(x.size(0)):
                    save_video_dir = os.path.join(self.opts.exp_dir, frame[i])
                    os.makedirs(save_video_dir, exist_ok=True)
                    save_video_path = os.path.join(save_video_dir, "video.mp4")
                    video_out = imageio.get_writer(save_video_path, mode='I', fps=10, codec='libx264')
                    
                    for frame_idx in range(len(self.render_poses)):
                        
                        # res = torch.cat((warp_imgs[frame_idx][i], masks[frame_idx][i].repeat(3, 1, 1), inps[frame_idx][i]), dim=2)
                        res = torch.cat((x[i], inps[frame_idx][i]), dim=2)
                        res = add_grid_lines(res, 512)
                        # res = inps[frame_idx][i]
                        res = self.tensor2im(res)

                        save_frame_dir = os.path.join(save_video_dir, 'frames')
                        os.makedirs(save_frame_dir, exist_ok=True)
                        save_frame_path = os.path.join(save_frame_dir, f'{frame_idx}.png')
                        cv2.imwrite(save_frame_path, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
                        video_out.append_data(res)

                    video_out.close()


            else:
                raise NotImplementedError(f"Novel view {self.opts.novel_view} not implemented")

        print("Inference done.")
        print("Results saved in:", self.opts.exp_dir)


    def forward_single(self, x_256, x, c, x_mirror_256, x_mirror, c_mirror, conf_map_mirror, depth_est, depth_est_mirror, frame, c_novel):

        with torch.no_grad():
            w = self.gan.encoder_forward(x_256)
            outs = self.gan.decoder.synthesis(w, c, noise_mode='const')
            outs_novel = self.gan.decoder.synthesis(w, c_novel, noise_mode='const')
        
        # y_hat = outs["image"]
        y_hat_novel = outs_novel["image"]

        res, res_mask, res_warp = None, None, None

        depth, depth_novel = outs["image_depth"], outs_novel["image_depth"]

        # Save the output
        if self.opts.infer_type == 'inv':

            res = y_hat_novel


        elif self.opts.infer_type == 'inpaint' or self.opts.infer_type == 'coarse' or self.opts.infer_type == 'warp':

            if self.use_depth_est:
                depth = F.interpolate(depth, size=(512, 512), mode='bicubic', align_corners=False)
                depth_est_align = calibrate_disparity(depth_est.squeeze(), depth.squeeze()).unsqueeze(1)
                # depth_est_align = self.ailgn_depth(depth_est, depth)
                
                # depth_est_mirror_align = calibrate_disparity(depth_est_mirror.squeeze(), depth.squeeze()).unsqueeze(1)
                depth_est_mirror_align = torch.flip(depth_est_align, dims=[3])

            _x = (x + 1) / 2  # -1~1 -> 0~1
            warp_depth = depth if not self.use_depth_est else depth_est_align
            warp_img, visable_mask, _ = self.warper.forward_warp(img1=_x, depth1=warp_depth, c1=c, c2=c_novel)
            warp_img = (warp_img * 2 - 1).clamp(-1, 1)  # 0~1 -> -1~1

            if self.opts.infer_type == 'warp':

                res = None
                res_mask = visable_mask
                res_warp = warp_img

            else:

                if self.opts.generator.input_mirror:
                    # with torch.no_grad():
                    #     outs_mirror = self.gan.decoder.synthesis(w, c_mirror, noise_mode='const')
                    depth_mirror = torch.flip(depth, dims=[3])
                    
                    _x_mirror = (x_mirror + 1) / 2
                    warp_depth_mirror = depth_mirror if not self.use_depth_est else depth_est_mirror_align
                    warp_img_mirror, visable_mask_mirror, _ = self.warper.forward_warp(img1=_x_mirror, depth1=warp_depth_mirror, c1=c_mirror, c2=c_novel)
                    warp_img_mirror = (warp_img_mirror * 2 - 1).clamp(-1, 1)  # 0~1 -> -1~1
                else:
                    warp_img_mirror, visable_mask_mirror = None, torch.ones_like(visable_mask)
                
                batch_inp = {
                    'image': warp_img, 'inversion': y_hat_novel,
                    'mask': 1 - visable_mask, 'ws': w,
                    'image_mirror': warp_img_mirror, 'mask_mirror': 1 - visable_mask_mirror,
                }
                inp, mask = self.inpaint(batch_inp)

                res = inp
                res_mask = mask
                res_warp = warp_img
        
        else:
            raise NotImplementedError(f"Infer Type {self.opts.infer_type} not implemented")
        

        return res, res_mask, res_warp, depth

    
    def get_ffhq_camera_params(self):
        with open(os.path.join(dataset_paths['train'], self.cam_json), 'r') as f:
            ffhq_cam_list = json.load(f)['labels']
        
        ffhq_cam_list = [[x[1]] for x in ffhq_cam_list]
        return ffhq_cam_list

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

    def tensor2im(self, var, norm=False):
        var = var.cpu().detach().permute(1, 2, 0).numpy()
        if norm:
            var = (var - var.min()) / (var.max() - var.min()) * 255
        else:
            var = ((var + 1) / 2)
            var = np.clip(var, 0, 1)
            var = var * 255
        return var.astype('uint8')

    def tensor2im01(self, var, norm=False):
        var = var.cpu().detach().permute(1, 2, 0).numpy()
        if norm:
            var = (var - var.min()) / (var.max() - var.min()) * 255
        else:
            # var = ((var + 1) / 2)
            var = np.clip(var, 0, 1)
            var = var * 255
        return var.astype('uint8')


@hydra.main(config_path="../configs", config_name="infer")
def main(opts: DictConfig):

    infer = Inference(opts)
    infer.forward()


if __name__ == '__main__':
    main()