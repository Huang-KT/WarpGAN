import os
import json
import random
import numpy as np
import cv2
from omegaconf import OmegaConf
from PIL import Image
import pytz
import datetime
import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as transforms
import kornia.morphology as km

from datasets.dataset_inpainting_static import ImageFolderDataset
from datasets.dataset_inpainting_synth_static import SynthImageFolderDataset
from configs.paths_config import dataset_static_paths as dataset_paths
# from configs.paths_config import dataset_cat_static_paths as dataset_paths
from configs.paths_config import model_paths

from models.wplusnet import WplusNet
from models.stylegan2.stylegan_ada import Discriminator
from models.eg3d.camera_utils import LookAtPoseSampler

######################################################################################
from models.saicinpainting.training.modules import make_generator, make_discriminator
from models.saicinpainting.training.losses.adversarial import make_discrim_loss
from models.saicinpainting.utils import set_requires_grad, add_prefix_to_keys, add_suffix_to_keys
from models.saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss, masked_l2_loss
from models.saicinpainting.training.losses.perceptual import PerceptualLoss, ResNetPL
######################################################################################

from training.ranger import Ranger
from criteria import id_loss, moco_loss
from criteria.lpips.lpips import LPIPS
from utils import common, train_utils
from utils.depth_utils import calibrate_disparity
# from utils.warp.rotate import rotate
from utils.warp.Splatting import Warper
from utils.common import get_time, add_grid_lines


EPS = 1


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0
        self.device = self.opts.device

        if self.opts.log.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)


        # Initialize network
        self.inpaintor = make_generator(self.opts, **self.opts.generator).to(self.device)
        self.discriminator = make_discriminator(**self.opts.discriminator).to(self.device)
        self.adversarial_loss = make_discrim_loss(**self.opts.losses.adversarial)

        # Initialize warper
        self.warper = Warper()

        # assert self.opts.losses.mirror.weight < 1e-5 or not self.opts.generator.cat_mirror, "Mirror loss is not supported with cat_mirror=True"

        # Initialize loss functions
        if self.opts.losses.perceptual.weight > 1e-5:
            self.loss_pl = PerceptualLoss().to(self.device).eval()

        if self.opts.losses.resnet_pl.weight > 1e-5:
            self.loss_resnet_pl = ResNetPL(**self.opts.losses.resnet_pl).to(self.device).eval()
        
        if self.opts.losses.lpips.weight > 1e-5:
            self.loss_lpips = LPIPS(net_type='alex').to(self.device).eval()

        if self.opts.losses.id.weight > 1e-5:
            if "ffhq" in dataset_paths['train'].lower() or "celeba" in dataset_paths['train'].lower():
                self.loss_id = id_loss.IDLoss().to(self.device).eval()
            else:
                self.loss_id = moco_loss.MocoLoss().to(self.device).eval()

        if self.opts.losses.latent.weight > 1e-5:
            self.gan = WplusNet(self.opts.gan).to(self.device).eval()
            set_requires_grad(self.gan, False)


        # Initialize optimizer
        self.optimizer_inpaintor = self.configure_inpaintor_optimizers()
        self.optimizer_discriminator = self.configure_discriminator_optimizers()


        # self.gan.psp_encoder.requires_grad_(False)
        set_requires_grad(self.inpaintor, False)
        set_requires_grad(self.discriminator, False)

        
        # Initialize dataset
        if not self.opts.data.add_LPFF and not self.opts.data.rebalanced:
            self.dataset_json = 'dataset.json'
        elif not self.opts.data.add_LPFF and self.opts.data.rebalanced:
            self.dataset_json = 'dataset_rebalanced.json'
        elif self.opts.data.add_LPFF and not self.opts.data.rebalanced:
            self.dataset_json = 'dataset_plus.json'
        else:
            self.dataset_json = 'dataset_plus_rebalanced.json'
        print("Dataset json: ", self.dataset_json)
        
        self.configure_datasets()
        self.configure_dataloaders()


        # Initialize logger
        log_dir = os.path.join(self.opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)


        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.log.save_interval is None:
            self.opts.log.save_interval = self.opts.max_steps

        
        # Resume training process from checkpoint path
        if self.opts.checkpoint_path is not None:
            self.resume()
        
        self.save_train_dir = os.path.join(self.logger.log_dir, 'images/train')
        self.save_val_dir = os.path.join(self.logger.log_dir, 'images/val')
        os.makedirs(self.save_train_dir, exist_ok=False)
        os.makedirs(self.save_val_dir, exist_ok=False)


    def inpaintor_forward(self, batch_inp):
        img, inversion, mask = batch_inp['image'], batch_inp['inversion'], batch_inp['mask']
        ws = batch_inp['ws']
        
        if self.opts.generator.input_mirror == 'cat' or self.opts.generator.input_mirror == 'condition':
            inp, masked_img, mask = self.get_inp(img, inversion, mask, self.opts.generator.cat_inv)
            img_mirror, mask_mirror = batch_inp['image_mirror'], batch_inp['mask_mirror']
            inp_mirror, masked_img_mirror, mask_mirror = self.get_inp(img_mirror, inversion, mask_mirror, cat_inv=self.opts.generator.cat_inv and self.opts.generator.input_mirror=='condition')
            inp = torch.cat([inp, inp_mirror], dim=1)
        elif self.opts.generator.input_mirror == 'condition_twice':
            img_mirror, mask_mirror = batch_inp['image_mirror'], batch_inp['mask_mirror']
            masked_img_mirror, mask_mirror = img_mirror, mask_mirror
            masked_img = img * (1 - mask)
            inp = torch.cat([masked_img, mask, inversion, torch.zeros_like(mask), masked_img_mirror, mask_mirror], dim=1)
        else:
            inp, masked_img, mask = self.get_inp(img, inversion, mask, self.opts.generator.cat_inv)
            masked_img_mirror, mask_mirror = None, None
        
        if self.opts.generator.kind == 'ffc_resnet':
            pred = self.inpaintor(inp)
        elif self.opts.generator.kind == 'ffc_style_resnet':
            pred = self.inpaintor(inp, ws)
        else:
            raise NotImplementedError(f"Generator kind {self.opts.generator.kind} not implemented")

        return pred, masked_img, mask, masked_img_mirror, mask_mirror

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

    def forward(self, batch, use_inv=True):

        x, x_256, c = batch['x'], batch['x_256'], batch['c']
        codes, y_hat, depth = batch['codes'], batch['y_hat'], batch['depth']
        c_novel, y_hat_novel, depth_novel = batch['c_novel'], batch['y_hat_novel'], batch['depth_novel']
        warp_ori = batch['warp_ori']

        if self.opts.generator.input_mirror:
            x_mirror, x_mirror_256, c_mirror = batch['x_mirror'], batch['x_mirror_256'], batch['c_mirror']
            depth_mirror = batch['depth_mirror']

        
        y_hat_resized = F.adaptive_avg_pool2d(y_hat, (256, 256))
        y_hat_novel_resized = F.adaptive_avg_pool2d(y_hat_novel, (256, 256))

        ## Warp image
        warp_img, visable_mask, _ = self.warper.forward_warp(img1=x if warp_ori else x_256, depth1=depth, c1=c, c2=c_novel)

        if self.opts.generator.input_mirror:
            warp_img_mirror, visable_mask_mirror, _ = self.warper.forward_warp(img1=x_mirror if warp_ori else x_mirror_256, depth1=depth_mirror, c1=c_mirror, c2=c_novel)
        else:
            warp_img_mirror, visable_mask_mirror = None, torch.ones_like(visable_mask)

        batch = dict(
            image=warp_img, image_mirror=warp_img_mirror,
            inversion=y_hat_novel if warp_ori else y_hat_novel_resized,
            mask=1-visable_mask, mask_mirror=1-visable_mask_mirror,
            ws=codes,
        )
        # Inpainting Warp Image
        pred, warp_img, mask, warp_img_mirror, mask_mirror = self.inpaintor_forward(batch)

        if use_inv is False:
            return {
                "x": x,
                "c": c, "c_novel": c_novel,
                "y_hat": y_hat, "y_hat_novel": y_hat_novel,
                "y_hat_resized": y_hat_resized, "y_hat_novel_resized": y_hat_novel_resized,
                'depth': depth, 'depth_novel': depth_novel,
                "warp_img": warp_img, "warp_img_mirror": warp_img_mirror,
                "mask": mask, "mask_mirror": mask_mirror,
                "pred_novel": pred
            }


        # Inverse Warp Image
        warp_warp_img, visable_mask_inv, _ = self.warper.forward_warp(img1=warp_img, depth1=depth_novel, c1=c_novel, c2=c)

        if self.opts.generator.input_mirror:
            warp_img_mirror_2 = torch.flip(warp_img, dims=[3])
            depth_novel_mirror = torch.flip(depth_novel, dims=[3])
            c_novel_mirror = self.get_mirror_c(c_novel)
            warp_warp_img_mirror, visable_mask_inv_mirror, _ = self.warper.forward_warp(img1=warp_img_mirror_2, depth1=depth_novel_mirror, c1=c_novel_mirror, c2=c)
        else:
            warp_warp_img_mirror, visable_mask_inv_mirror = None, torch.ones_like(visable_mask_inv)

        batch_inv_warp = dict(
            image=warp_warp_img, image_mirror=warp_warp_img_mirror,
            inversion=y_hat if warp_ori else y_hat_resized,
            mask=1-visable_mask_inv, mask_mirror=1-visable_mask_inv_mirror,
            ws=codes,
        )
        # Inpainting Inverse Warp Image
        pred_inv_warp, warp_warp_img, mask_inv, warp_warp_img_mirror, mask_inv_mirror = self.inpaintor_forward(batch_inv_warp)


        # Inpaintings Inverse Predicted Image
        if self.opts.warp.warp_pred:
            warp_pred, _, _ = self.warper.forward_warp(img1=pred, depth1=depth_novel, c1=c_novel, c2=c)

            if self.opts.generator.input_mirror:
                pred_mirror = torch.flip(pred, dims=[3])
                warp_pred_mirror, _, _ = self.warper.forward_warp(img1=pred_mirror, depth1=depth_novel_mirror, c1=c_novel_mirror, c2=c)
            else:
                warp_pred_mirror = None

            batch_inv_pred = dict(
                image=warp_pred, image_mirror=warp_pred_mirror,
                inversion=y_hat if warp_ori else y_hat_resized,
                mask=1-visable_mask_inv, mask_mirror=1-visable_mask_inv_mirror,
                ws=codes,
            )
            pred_inv_pred, warp_pred, _, warp_pred_mirror, _ = self.inpaintor_forward(batch_inv_pred)
        else:
            warp_pred, pred_inv_pred, warp_pred_mirror = None, None, None

        return {
            "x": x,
            "c": c, "c_novel": c_novel,
            "y_hat": y_hat, "y_hat_novel": y_hat_novel,
            "y_hat_resized": y_hat_resized, "y_hat_novel_resized": y_hat_novel_resized,
            'depth': depth, 'depth_novel': depth_novel,
            "warp_img": warp_img, "warp_img_mirror": warp_img_mirror,
            "warp_warp_img": warp_warp_img, "warp_warp_img_mirror": warp_warp_img_mirror,
            "warp_pred": warp_pred, "warp_pred_mirror": warp_pred_mirror,
            "mask": mask, "mask_mirror": mask_mirror,
            "mask_inv": mask_inv, "mask_inv_mirror": mask_inv_mirror,
            "pred_novel": pred, "pred_inv_warp": pred_inv_warp, "pred_inv_pred": pred_inv_pred,
        }


    def train(self):

        print("Start training with static dataset...")

        self.inpaintor.train()
        self.discriminator.train()
        torch.cuda.empty_cache()

        real_iterator = iter(self.train_dataloader)
        if self.opts.data.synth.able:
            synth_iterator = iter(self.train_synth_dataloader)


        while self.global_step <= self.opts.max_steps:

            try:
                batch_data_real = next(real_iterator)
            except StopIteration:
                print("End of real_iterator reached. Resetting real_iterator.")
                real_iterator = iter(self.train_dataloader)
                batch_data_real = next(real_iterator)

            x_256, x, c, codes, y_hat, depth = batch_data_real[:6]
            x_mirror_256, x_mirror, c_mirror, y_hat_mirror, depth_mirror, conf_map_mirror = batch_data_real[6:12]
            c_novel, y_hat_novel, depth_novel = batch_data_real[12:15]

            x_256, x, c = x_256.to(self.device).float(), x.to(self.device).float(), c.to(self.device).float()
            codes = codes.to(self.device).float()
            y_hat, depth = y_hat.to(self.device).float(), depth.to(self.device).float()
            x_mirror_256, x_mirror, c_mirror = x_mirror_256.to(self.device).float(), x_mirror.to(self.device).float(), c_mirror.to(self.device).float()
            y_hat_mirror, depth_mirror = y_hat_mirror.to(self.device).float(), depth_mirror.to(self.device).float()
            if self.opts.losses.mirror.weight > 1e-5:
                conf_map_mirror = conf_map_mirror.to(self.device).float()
            c_novel, y_hat_novel, depth_novel = c_novel.to(self.device).float(), y_hat_novel.to(self.device).float(), depth_novel.to(self.device).float()

            loss_dict = {}

            # Update G
            set_requires_grad(self.inpaintor, True)
            set_requires_grad(self.discriminator, False)


            batch_real = {
                'x': x, 'x_256': x_256, 'c': c,
                'codes': codes, 'y_hat': y_hat, 'depth': depth,
                'c_novel': c_novel, 'y_hat_novel': y_hat_novel, 'depth_novel': depth_novel,
                'warp_ori': self.opts.warp.warp_ori_train,
                'x_mirror': x_mirror, 'x_mirror_256': x_mirror_256, 'c_mirror': c_mirror,
                'depth_mirror': depth_mirror,
            }
            batch_inp = self.forward(batch_real)
            
            loss_inpaintor, metrics_inpaintor = self.cal_inpaintor_loss({
                'image': x if self.opts.warp.warp_ori_train else x_256,
                'c': c, 'c_novel': c_novel, 'codes': codes,
                'depth': depth, 'depth_novel': depth_novel,
                'mask_inv': batch_inp['mask_inv'],
                'pred_novel': batch_inp['pred_novel'],
                'pred_inv_warp': batch_inp['pred_inv_warp'], 'pred_inv_pred': batch_inp['pred_inv_pred'],
            })


            # if self.opts.losses.mirror.weight > 1e-5:
            #     batch_mirror = {
            #         'x': x.clone().detach(), 'x_256': x_256.clone().detach(), 'c': c.clone().detach(),
            #         'codes': codes.clone().detach(), 'y_hat': y_hat.clone().detach(), 'depth': depth.clone().detach(),
            #         'c_novel': c_mirror, 'y_hat_novel': y_hat_mirror, 'depth_novel': depth_mirror,
            #         'warp_ori': self.opts.warp.warp_ori_train,
            #         'x_mirror': x_mirror, 'x_mirror_256': x_mirror_256, 'c_mirror': c_mirror,
            #         'depth_mirror': depth_mirror,
            #     }
            #     batch_inp_mirror = self.forward(batch_mirror, use_inv=False)

            #     # if self.opts.losses.mirror.rec_only:
            #     loss_inpaintor_mirror, metrics_inpaintor_mirror = self.cal_inpaintor_mirror_loss(
            #         {
            #             'image_mirror': x_mirror if self.opts.warp.warp_ori_train else x_mirror_256,
            #             'mask': batch_inp_mirror['mask'],
            #             'pred_mirror': batch_inp_mirror['pred_novel'],
            #             'codes': codes.clone().detach(),
            #         },
            #         conf_map=conf_map_mirror,
            #         rec_only=True
            #     )

            #     loss_inpaintor = loss_inpaintor + loss_inpaintor_mirror * self.opts.losses.mirror.weight
            #     metrics_inpaintor = {**metrics_inpaintor, **metrics_inpaintor_mirror}                
            # metrics_inpaintor['gen_loss_real'] = float(loss_inpaintor)
            
            
            self.optimizer_inpaintor.zero_grad()
            loss_inpaintor.backward()
            self.optimizer_inpaintor.step()
            self.optimizer_inpaintor.zero_grad()

            loss_dict = {**metrics_inpaintor}


            # Update D
            if self.opts.losses.adversarial.weight > 1e-5:

                set_requires_grad(self.inpaintor, False)
                set_requires_grad(self.discriminator, True)


                # if self.opts.losses.mirror.weight > 1e-5:

                if batch_inp['pred_inv_warp'] is None:
                    batch_disc = {
                        'image': (x if self.opts.warp.warp_ori_train else x_256).clone().detach(),
                        'pred' : batch_inp['pred_novel'].clone().detach(),
                        'mask' : batch_inp['mask'].clone().detach(),
                    }
                elif batch_inp['pred_inv_pred'] is None:
                    batch_disc = {
                        'image': (x if self.opts.warp.warp_ori_train else x_256).clone().detach().repeat(2, 1, 1, 1),
                        'pred' : torch.cat([batch_inp['pred_novel'], batch_inp['pred_inv_warp']], dim=0).clone().detach(),
                        'mask' : torch.cat([batch_inp['mask'], batch_inp['mask_inv']], dim=0).clone().detach(),
                    }
                else:
                    batch_disc = {
                        'image': (x if self.opts.warp.warp_ori_train else x_256).clone().detach().repeat(3, 1, 1, 1),
                        'pred' : torch.cat([batch_inp['pred_novel'], batch_inp['pred_inv_warp'], batch_inp['pred_inv_pred']], dim=0).clone().detach(),
                        'mask' : torch.cat([batch_inp['mask'], batch_inp['mask_inv'], batch_inp['mask_inv']], dim=0).clone().detach(),
                    }

                loss_discriminator, metrics_discriminator = self.cal_discriminator_loss(batch_disc)
                
                self.optimizer_discriminator.zero_grad()
                loss_discriminator.backward()
                self.optimizer_discriminator.step()
                self.optimizer_discriminator.zero_grad()

                loss_dict = {**loss_dict, **metrics_discriminator}



            if self.opts.losses.mirror.weight > 1e-5:

                set_requires_grad(self.inpaintor, True)
                set_requires_grad(self.discriminator, False)

                batch_mirror = {
                    'x': x.clone().detach(), 'x_256': x_256.clone().detach(), 'c': c.clone().detach(),
                    'codes': codes.clone().detach(), 'y_hat': y_hat.clone().detach(), 'depth': depth.clone().detach(),
                    'c_novel': c_mirror, 'y_hat_novel': y_hat_mirror, 'depth_novel': depth_mirror,
                    'warp_ori': self.opts.warp.warp_ori_train,
                    'x_mirror': x_mirror, 'x_mirror_256': x_mirror_256, 'c_mirror': c_mirror,
                    'depth_mirror': depth_mirror,
                }
                batch_inp_mirror = self.forward(batch_mirror, use_inv=False)

                # if self.opts.losses.mirror.rec_only:
                loss_inpaintor_mirror, metrics_inpaintor_mirror = self.cal_inpaintor_mirror_loss(
                    {
                        'image_mirror': x_mirror if self.opts.warp.warp_ori_train else x_mirror_256,
                        'mask': batch_inp_mirror['mask'],
                        'pred_mirror': batch_inp_mirror['pred_novel'],
                        'codes': codes.clone().detach(),
                    },
                    conf_map=conf_map_mirror,
                    rec_only=True
                )

                loss_inpaintor_mirror = loss_inpaintor_mirror * self.opts.losses.mirror.weight
                # metrics_inpaintor = {**metrics_inpaintor, **metrics_inpaintor_mirror}

                self.optimizer_inpaintor.zero_grad()
                loss_inpaintor_mirror.backward()
                self.optimizer_inpaintor.step()
                self.optimizer_inpaintor.zero_grad()

                loss_dict = {**loss_dict, **metrics_inpaintor_mirror}
            
            else:
                loss_inpaintor_mirror = 0.0
            
            loss_dict['gen_loss_real'] = float(loss_inpaintor + loss_inpaintor_mirror)



            if self.opts.data.synth.able:

                try:
                    batch_data_synth = next(synth_iterator)
                except StopIteration:
                    print("End of synth_iterator reached. Resetting synth_iterator.")
                    synth_iterator = iter(self.train_synth_dataloader)
                    batch_data_synth = next(synth_iterator)

                src_img_256, src_img, src_c, codes_synth, src_hat, src_depth = batch_data_synth[:6]
                target_img_256, target_img, target_c, target_hat, target_depth = batch_data_synth[6:11]

                src_img_256, src_img, src_c = src_img_256.to(self.device).float(), src_img.to(self.device).float(), src_c.to(self.device).float()
                codes_synth = codes_synth.to(self.device).float()
                src_hat, src_depth = src_hat.to(self.device).float(), src_depth.to(self.device).float()
                target_img_256, target_img, target_c = target_img_256.to(self.device).float(), target_img.to(self.device).float(), target_c.to(self.device).float()
                target_hat, target_depth = target_hat.to(self.device).float(), target_depth.to(self.device).float()


                # Update G
                set_requires_grad(self.inpaintor, True)
                set_requires_grad(self.discriminator, False)

                if self.opts.generator.input_mirror:
                    src_img_mirror = torch.flip(src_img, dims=[3])
                    src_img_mirror_256 = torch.flip(src_img_256, dims=[3])
                    src_c_mirror = self.get_mirror_c(src_c)
                    src_depth_mirror = torch.flip(src_depth, dims=[3])
                else:
                    src_img_mirror, src_img_mirror_256, src_c_mirror, src_depth_mirror = None, None, None, None

                batch_synth = {
                    'x': src_img, 'x_256': src_img_256, 'c': src_c,
                    'codes': codes_synth, 'y_hat': src_hat, 'depth': src_depth,
                    'c_novel': target_c, 'y_hat_novel': target_hat, 'depth_novel': target_depth,
                    'warp_ori': self.opts.warp.warp_ori_train,
                    'x_mirror': src_img_mirror, 'x_mirror_256': src_img_mirror_256, 'c_mirror': src_c_mirror,
                    'depth_mirror': src_depth_mirror,
                }
                
                batch_inp_synth = self.forward(batch_synth, use_inv=False)
                batch_rec_synth = {
                    'image': target_img if self.opts.warp.warp_ori_train else target_img_256,
                    'mask': batch_inp_synth['mask'],
                    'pred': batch_inp_synth['pred_novel'],
                    'codes': codes_synth,
                }
                loss_inpaintor_synth, metrics_inpaintor_synth = self.cal_inpaintor_rec_loss(batch_rec_synth, conf_map=None)
                metrics_inpaintor_synth['gen_loss'] = loss_inpaintor_synth
                metrics_inpaintor_synth = {k: float(v) for k, v in metrics_inpaintor_synth.items()}

                self.optimizer_inpaintor.zero_grad()
                loss_inpaintor_synth.backward()
                self.optimizer_inpaintor.step()
                self.optimizer_inpaintor.zero_grad()

                loss_dict_synth = {**metrics_inpaintor_synth}


                # Update D
                if self.opts.losses.adversarial.weight > 1e-5:

                    set_requires_grad(self.inpaintor, False)
                    set_requires_grad(self.discriminator, True)

                    batch_disc_synth = {
                        'image': (src_img if self.opts.warp.warp_ori_train else src_img_256).clone().detach(),
                        'pred' : batch_inp_synth['pred_novel'].clone().detach(),
                        'mask' : batch_inp_synth['mask'].clone().detach(),
                    }
                    loss_discriminator_synth, metrics_discriminator_synth = self.cal_discriminator_loss(batch_disc_synth)

                    self.optimizer_discriminator.zero_grad()
                    loss_discriminator_synth.backward()
                    self.optimizer_discriminator.step()
                    self.optimizer_discriminator.zero_grad()

                    loss_dict_synth = {**loss_dict_synth, **metrics_discriminator_synth}

                loss_dict.update(add_suffix_to_keys(loss_dict_synth, '_synth'))


            if self.global_step % self.opts.log.image_interval == 0 or (self.global_step <= 1000 and self.global_step % 25 == 0):
                
                # print('Saving images...')

                y_hat, y_hat_novel = batch_inp['y_hat'], batch_inp['y_hat_novel']
                y_hat_resized, y_hat_novel_resized = batch_inp['y_hat_resized'], batch_inp['y_hat_novel_resized']
                warp_img, warp_warp_img = batch_inp['warp_img'].cpu().detach(), batch_inp['warp_warp_img'].cpu().detach()
                mask, mask_inv = batch_inp['mask'].cpu().detach(), batch_inp['mask_inv'].cpu().detach()
                pred_novel, pred_inv_warp = batch_inp['pred_novel'].cpu().detach(), batch_inp['pred_inv_warp'].cpu().detach()
                
                _x = (x if self.opts.warp.warp_ori_train else x_256).cpu().detach()
                _y_hat = (y_hat if self.opts.warp.warp_ori_train else y_hat_resized).cpu().detach()
                _y_hat_novel = (y_hat_novel if self.opts.warp.warp_ori_train else y_hat_novel_resized).cpu().detach()

                fake_novel = (warp_img * (1 - mask) + mask * _y_hat_novel).cpu().detach()
                fake_inv_warp = (warp_warp_img * (1 - mask_inv) + mask_inv * _y_hat).cpu().detach()

                _blank = torch.zeros_like(_x[0])
                res_x = torch.cat([_x[0], _blank], dim=1)
                res_y = torch.cat([_y_hat_novel[0], _y_hat[0]], dim=1)
                res_warp = torch.cat([warp_img[0], warp_warp_img[0]], dim=1)
                res_mask = torch.cat([mask[0], mask_inv[0]], dim=1).expand(3, -1, -1)
                res_fake = torch.cat([fake_novel[0], fake_inv_warp[0]], dim=1)
                res_pred = torch.cat([pred_novel[0], pred_inv_warp[0]], dim=1)

                if self.opts.generator.input_mirror:
                    warp_img_mirror = batch_inp['warp_img_mirror'].cpu().detach()
                    warp_warp_img_mirror = batch_inp['warp_warp_img_mirror'].cpu().detach()
                    mask_mirror = batch_inp['mask_mirror'].cpu().detach()
                    mask_inv_mirror = batch_inp['mask_inv_mirror'].cpu().detach()
                    res_warp_mirror = torch.cat([warp_img_mirror[0], warp_warp_img_mirror[0]], dim=1)
                    res_mask_mirror = torch.cat([mask_mirror[0], mask_inv_mirror[0]], dim=1).expand(3, -1, -1)
                    res = torch.cat([res_x, res_y, res_warp, res_mask, res_fake, res_warp_mirror, res_mask_mirror, res_pred], dim=2)
                else:
                    res = torch.cat([res_x, res_y, res_warp, res_mask, res_fake, res_pred], dim=2)

                if self.opts.warp.warp_pred:
                    warp_pred = batch_inp['warp_pred'].cpu().detach()
                    pred_inv_pred = batch_inp['pred_inv_pred'].cpu().detach()
                    fake_inv_pred = warp_pred * (1 - mask_inv) + mask_inv * _y_hat

                    if self.opts.generator.input_mirror:
                        warp_pred_mirror = batch_inp['warp_pred_mirror'].cpu().detach()
                        res_warp_pred = torch.cat([_blank, _y_hat[0], warp_pred[0], mask_inv[0].expand(3, -1, -1), fake_inv_pred[0], warp_pred_mirror[0], mask_inv_mirror[0].expand(3, -1, -1), pred_inv_pred[0]], dim=2)
                    else:
                        res_warp_pred = torch.cat([_blank, _y_hat[0], warp_pred[0], mask_inv[0].expand(3, -1, -1), fake_inv_pred[0], pred_inv_pred[0]], dim=2)
                    res = torch.cat([res, res_warp_pred], dim=1)

                if self.opts.losses.mirror.weight > 1e-5:
                    mask_mirror = batch_inp_mirror['mask'].cpu().detach()
                    warp_img_mirror = batch_inp_mirror['warp_img'].cpu().detach()
                    y_hat_mirror, y_hat_mirror_resized = batch_inp_mirror['y_hat_novel'], batch_inp_mirror['y_hat_novel_resized']
                    pred_mirror = batch_inp_mirror['pred_novel'].cpu().detach()

                    _x_mirror = (x_mirror if self.opts.warp.warp_ori_train else x_mirror_256).cpu().detach()
                    _y_hat_mirror = (y_hat_mirror if self.opts.warp.warp_ori_train else y_hat_mirror_resized).cpu().detach()
                    
                    fake_mirror = (warp_img_mirror * (1 - mask_mirror) + mask_mirror * _y_hat_mirror).cpu().detach()
                    
                    if self.opts.generator.input_mirror:
                        mirror_warp_img_mirror = batch_inp_mirror['warp_img_mirror'].cpu().detach()
                        mirror_mask_mirror = batch_inp_mirror['mask_mirror'].cpu().detach()
                        res_mirror = torch.cat([_x_mirror[0], _y_hat_mirror[0], warp_img_mirror[0], mask_mirror[0].expand(3, -1, -1), fake_mirror[0], mirror_warp_img_mirror[0], mirror_mask_mirror[0].expand(3, -1, -1), pred_mirror[0]], dim=2)
                    else:
                        res_mirror = torch.cat([_x_mirror[0], _y_hat_mirror[0], warp_img_mirror[0], mask_mirror[0].expand(3, -1, -1), fake_mirror[0], pred_mirror[0]], dim=2)
                    res = torch.cat([res, res_mirror], dim=1)

                if self.opts.data.synth.able:

                    mask_synth = batch_inp_synth['mask'].cpu().detach()
                    warp_img_synth = batch_inp_synth['warp_img'].cpu().detach()
                    y_hat_synth, y_hat_synth_resized = batch_inp_synth['y_hat_novel'], batch_inp_synth['y_hat_novel_resized']
                    pred_synth = batch_inp_synth['pred_novel'].cpu().detach()

                    _src_img = (src_img if self.opts.warp.warp_ori_train else src_img_256).cpu().detach()
                    _target_img = (target_img if self.opts.warp.warp_ori_train else target_img_256).cpu().detach()
                    _y_hat_synth = (y_hat_synth if self.opts.warp.warp_ori_train else y_hat_synth_resized).cpu().detach()
                    
                    fake_synth = warp_img_synth * (1 - mask_synth) + mask_synth * _y_hat_synth

                    if self.opts.generator.input_mirror:
                        warp_img_synth_mirror = batch_inp_synth['warp_img_mirror'].cpu().detach()
                        mask_synth_mirror = batch_inp_synth['mask_mirror'].cpu().detach()
                        res_synth = torch.cat([_src_img[0], _y_hat_synth[0], warp_img_synth[0], mask_synth[0].expand(3, -1, -1), fake_synth[0], warp_img_synth_mirror[0], mask_synth_mirror[0].expand(3, -1, -1), pred_synth[0], _target_img[0]], dim=2)
                    else:
                        res_synth = torch.cat([_src_img[0], _y_hat_synth[0], warp_img_synth[0], mask_synth[0].expand(3, -1, -1), fake_synth[0], pred_synth[0], _target_img[0]], dim=2)
                    __blank = torch.cat([_blank, _blank], dim=1)
                    if self.opts.warp.warp_pred:
                        __blank = torch.cat([__blank, _blank], dim=1)
                    if self.opts.losses.mirror.weight > 1e-5:
                        __blank = torch.cat([__blank, _blank], dim=1)
                    res = torch.cat([res, __blank], dim=2)
                    res = torch.cat([res, res_synth], dim=1)

                res = add_grid_lines(res, 512 if self.opts.warp.warp_ori_train else 256)

                save_train_path = os.path.join(self.save_train_dir, f'{self.global_step:04d}.jpg')
                cv2.imwrite(save_train_path, cv2.cvtColor(self.tensor2im(res), cv2.COLOR_RGB2BGR))

                # print('Images saved.')


            if self.global_step % self.opts.log.board_interval == 0 or (self.global_step <= 100 and self.global_step % 25 == 0):
                self.print_metrics(loss_dict, prefix='train')
                self.log_metrics(loss_dict, prefix='train')

            # Log images of first batch to wandb
            # if self.opts.log.use_wandb and batch_idx == 0 and self.phase == "first":
            #     self.wb_logger.log_images_to_wandb(x, final_out, final_out_novel, id_logs, prefix="train", step=self.global_step, opts=self.opts)

            # Validation related
            val_loss_dict = None
            if self.global_step % self.opts.log.val_interval == 0 or self.global_step == self.opts.max_steps:
                # print('Validating...')
                val_loss_dict = self.validate()
                if val_loss_dict is not None:
                    new_val_loss = val_loss_dict['gen_loss']

                if val_loss_dict and (self.best_val_loss is None or new_val_loss < self.best_val_loss):
                    self.best_val_loss = new_val_loss
                    self.checkpoint_me(val_loss_dict, is_best=True)
                # print('Validation done.')

            if self.global_step % self.opts.log.save_interval == 0 or self.global_step == self.opts.max_steps:
                if val_loss_dict is not None:
                    self.checkpoint_me(val_loss_dict, is_best=False)
                else:
                    self.checkpoint_me(loss_dict, is_best=False)

            if self.global_step == self.opts.max_steps:
                # print('OMG, finished training!')
                break

            # print("Global Step: ", self.global_step)
            self.global_step += 1

        print('OMG, finished training!')

    def validate(self):
        self.inpaintor.eval()
        self.discriminator.eval()
        # torch.cuda.empty_cache()

        agg_loss_dict = []
        val_img_log_num = 100
        for batch_idx, batch in enumerate(self.test_dataloader):

            x_256, x, c, codes, y_hat, depth = batch[:6]
            x_mirror_256, x_mirror, c_mirror, y_hat_mirror, depth_mirror, conf_map_mirror = batch[6:12]
            c_novel, y_hat_novel, depth_novel = batch[12:15]

            x_256, x, c = x_256.to(self.device).float(), x.to(self.device).float(), c.to(self.device).float()
            codes = codes.to(self.device).float()
            y_hat, depth = y_hat.to(self.device).float(), depth.to(self.device).float()
            x_mirror_256, x_mirror, c_mirror = x_mirror_256.to(self.device).float(), x_mirror.to(self.device).float(), c_mirror.to(self.device).float()
            y_hat_mirror, depth_mirror = y_hat_mirror.to(self.device).float(), depth_mirror.to(self.device).float()
            c_novel, y_hat_novel, depth_novel = c_novel.to(self.device).float(), y_hat_novel.to(self.device).float(), depth_novel.to(self.device).float()
            if self.opts.losses.mirror.weight > 1e-5:
                conf_map_mirror = conf_map_mirror.to(self.device).float()

            with torch.no_grad():
                batch = {
                    'x': x, 'x_256': x_256, 'c': c,
                    'codes': codes, 'y_hat': y_hat, 'depth': depth,
                    'c_novel': c_novel, 'y_hat_novel': y_hat_novel, 'depth_novel': depth_novel,
                    'warp_ori': self.opts.warp.warp_ori_val,
                    'x_mirror': x_mirror, 'x_mirror_256': x_mirror_256, 'c_mirror': c_mirror,
                    'depth_mirror': depth_mirror,
                }
                batch_inp = self.forward(batch)
                loss_inpaintor, metrics_inpaintor = self.cal_inpaintor_loss({
                    'image': x if self.opts.warp.warp_ori_val else x_256,
                    'c': c, 'c_novel': batch_inp['c_novel'], 'codes': codes,
                    'depth': batch_inp['depth'], 'depth_novel': batch_inp['depth_novel'],
                    'mask_inv': batch_inp['mask_inv'],
                    'pred_novel': batch_inp['pred_novel'],
                    'pred_inv_warp': batch_inp['pred_inv_warp'], 'pred_inv_pred': batch_inp['pred_inv_pred'],
                })

                if self.opts.losses.mirror.weight > 1e-5:
                    batch_mirror = {
                        'x': x.clone().detach(), 'x_256': x_256.clone().detach(), 'c': c.clone().detach(),
                        'codes': codes.clone().detach(), 'y_hat': y_hat.clone().detach(), 'depth': depth.clone().detach(),
                        'c_novel': c_mirror, 'y_hat_novel': y_hat_mirror, 'depth_novel': depth_mirror,
                        'warp_ori': self.opts.warp.warp_ori_val,
                        'x_mirror': x_mirror, 'x_mirror_256': x_mirror_256, 'c_mirror': c_mirror,
                        'depth_mirror': depth_mirror,
                    }
                    batch_inp_mirror = self.forward(batch_mirror, use_inv=False)

                    # if self.opts.losses.mirror.rec_only:
                    loss_inpaintor_mirror, metrics_inpaintor_mirror = self.cal_inpaintor_mirror_loss({
                        'image_mirror': x_mirror if self.opts.warp.warp_ori_val else x_mirror_256,
                        'mask': batch_inp_mirror['mask'],
                        'pred_mirror': batch_inp_mirror['pred_novel'],
                        'codes': codes.clone().detach(),
                    },
                    conf_map=conf_map_mirror,
                    rec_only=True)

                    loss_inpaintor = loss_inpaintor + loss_inpaintor_mirror * self.opts.losses.mirror.weight
                    metrics_inpaintor = {**metrics_inpaintor, **metrics_inpaintor_mirror}

                metrics_inpaintor['gen_loss_real'] = float(loss_inpaintor)

            agg_loss_dict.append(metrics_inpaintor)
            


            # Logging related
            if batch_idx < val_img_log_num:

                y_hat, y_hat_novel = batch_inp['y_hat'], batch_inp['y_hat_novel']
                y_hat_resized, y_hat_novel_resized = batch_inp['y_hat_resized'], batch_inp['y_hat_novel_resized']
                warp_img, warp_warp_img = batch_inp['warp_img'].cpu().detach(), batch_inp['warp_warp_img'].cpu().detach()
                mask, mask_inv = batch_inp['mask'].cpu().detach(), batch_inp['mask_inv'].cpu().detach()
                pred_novel, pred_inv_warp = batch_inp['pred_novel'].cpu().detach(), batch_inp['pred_inv_warp'].cpu().detach()
                
                _x = (x if self.opts.warp.warp_ori_val else x_256).cpu().detach()
                _y_hat = (y_hat if self.opts.warp.warp_ori_val else y_hat_resized).cpu().detach()
                _y_hat_novel = (y_hat_novel if self.opts.warp.warp_ori_val else y_hat_novel_resized).cpu().detach()

                fake_novel = (warp_img * (1 - mask) + mask * _y_hat_novel).cpu().detach()
                fake_inv_warp = (warp_warp_img * (1 - mask_inv) + mask_inv * _y_hat).cpu().detach()

                _blank = torch.zeros_like(_x[0])
                res_x = torch.cat([_x[0], _blank], dim=1)
                res_y = torch.cat([_y_hat_novel[0], _y_hat[0]], dim=1)
                res_warp = torch.cat([warp_img[0], warp_warp_img[0]], dim=1)
                res_mask = torch.cat([mask[0], mask_inv[0]], dim=1).expand(3, -1, -1)
                res_fake = torch.cat([fake_novel[0], fake_inv_warp[0]], dim=1)
                res_pred = torch.cat([pred_novel[0], pred_inv_warp[0]], dim=1)

                if self.opts.generator.input_mirror:
                    warp_img_mirror = batch_inp['warp_img_mirror'].cpu().detach()
                    warp_warp_img_mirror = batch_inp['warp_warp_img_mirror'].cpu().detach()
                    mask_mirror = batch_inp['mask_mirror'].cpu().detach()
                    mask_inv_mirror = batch_inp['mask_inv_mirror'].cpu().detach()
                    res_warp_mirror = torch.cat([warp_img_mirror[0], warp_warp_img_mirror[0]], dim=1)
                    res_mask_mirror = torch.cat([mask_mirror[0], mask_inv_mirror[0]], dim=1).expand(3, -1, -1)
                    res = torch.cat([res_x, res_y, res_warp, res_mask, res_fake, res_warp_mirror, res_mask_mirror, res_pred], dim=2)
                else:
                    res = torch.cat([res_x, res_y, res_warp, res_mask, res_fake, res_pred], dim=2)

                if self.opts.warp.warp_pred:
                    warp_pred = batch_inp['warp_pred'].cpu().detach()
                    pred_inv_pred = batch_inp['pred_inv_pred'].cpu().detach()
                    fake_inv_pred = warp_pred * (1 - mask_inv) + mask_inv * _y_hat

                    if self.opts.generator.input_mirror:
                        warp_pred_mirror = batch_inp['warp_pred_mirror'].cpu().detach()
                        res_warp_pred = torch.cat([_blank, _y_hat[0], warp_pred[0], mask_inv[0].expand(3, -1, -1), fake_inv_pred[0], warp_pred_mirror[0], mask_inv_mirror[0].expand(3, -1, -1), pred_inv_pred[0]], dim=2)
                    else:
                        res_warp_pred = torch.cat([_blank, _y_hat[0], warp_pred[0], mask_inv[0].expand(3, -1, -1), fake_inv_pred[0], pred_inv_pred[0]], dim=2)
                    res = torch.cat([res, res_warp_pred], dim=1)

                if self.opts.losses.mirror.weight > 1e-5:
                    mask_mirror = batch_inp_mirror['mask'].cpu().detach()
                    warp_img_mirror = batch_inp_mirror['warp_img'].cpu().detach()
                    y_hat_mirror, y_hat_mirror_resized = batch_inp_mirror['y_hat_novel'], batch_inp_mirror['y_hat_novel_resized']
                    pred_mirror = batch_inp_mirror['pred_novel'].cpu().detach()

                    _x_mirror = (x_mirror if self.opts.warp.warp_ori_val else x_mirror_256).cpu().detach()
                    _y_hat_mirror = (y_hat_mirror if self.opts.warp.warp_ori_val else y_hat_mirror_resized).cpu().detach()
                    
                    fake_mirror = (warp_img_mirror * (1 - mask_mirror) + mask_mirror * _y_hat_mirror).cpu().detach()
                    
                    if self.opts.generator.input_mirror:
                        mirror_warp_img_mirror = batch_inp_mirror['warp_img_mirror'].cpu().detach()
                        mirror_mask_mirror = batch_inp_mirror['mask_mirror'].cpu().detach()
                        res_mirror = torch.cat([_x_mirror[0], _y_hat_mirror[0], warp_img_mirror[0], mask_mirror[0].expand(3, -1, -1), fake_mirror[0], mirror_warp_img_mirror[0], mirror_mask_mirror[0].expand(3, -1, -1), pred_mirror[0]], dim=2)
                    else:
                        res_mirror = torch.cat([_x_mirror[0], _y_hat_mirror[0], warp_img_mirror[0], mask_mirror[0].expand(3, -1, -1), fake_mirror[0], pred_mirror[0]], dim=2)
                    res = torch.cat([res, res_mirror], dim=1)

                res = add_grid_lines(res, 512 if self.opts.warp.warp_ori_val else 256)
                
                _save_val_dir = os.path.join(self.save_val_dir, f'{self.global_step:04d}')
                os.makedirs(_save_val_dir, exist_ok=True)
                save_val_path = os.path.join(_save_val_dir, f'{batch_idx:04d}.jpg')
                cv2.imwrite(save_val_path, cv2.cvtColor(self.tensor2im(res), cv2.COLOR_RGB2BGR))


            # Log images of first batch to wandb
            # if self.opts.log.use_wandb and batch_idx == 0:
            #     self.wb_logger.log_images_to_wandb(x, final_out, final_out_novel, id_logs, prefix="test", step=self.global_step, opts=self.opts)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 20:
                self.inpaintor.train()
                self.discriminator.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.inpaintor.train()
        self.discriminator.train()
        return loss_dict



    def infer(self):
        self.inpaintor.eval()
        self.discriminator.eval()
        # torch.cuda.empty_cache()

        val_img_log_num = 200
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, total=val_img_log_num)):

            x_256, x, c, codes, y_hat, depth = batch[:6]
            x_mirror_256, x_mirror, c_mirror, y_hat_mirror, depth_mirror, conf_map_mirror = batch[6:12]
            c_novel, y_hat_novel, depth_novel = batch[12:15]

            x_256, x, c = x_256.to(self.device).float(), x.to(self.device).float(), c.to(self.device).float()
            codes = codes.to(self.device).float()
            y_hat, depth = y_hat.to(self.device).float(), depth.to(self.device).float()
            x_mirror_256, x_mirror, c_mirror = x_mirror_256.to(self.device).float(), x_mirror.to(self.device).float(), c_mirror.to(self.device).float()
            y_hat_mirror, depth_mirror = y_hat_mirror.to(self.device).float(), depth_mirror.to(self.device).float()
            c_novel, y_hat_novel, depth_novel = c_novel.to(self.device).float(), y_hat_novel.to(self.device).float(), depth_novel.to(self.device).float()
            if self.opts.losses.mirror.weight > 1e-5:
                conf_map_mirror = conf_map_mirror.to(self.device).float()

            with torch.no_grad():
                batch = {
                    'x': x, 'x_256': x_256, 'c': c,
                    'codes': codes, 'y_hat': y_hat, 'depth': depth,
                    'c_novel': c_novel, 'y_hat_novel': y_hat_novel, 'depth_novel': depth_novel,
                    'warp_ori': self.opts.warp.warp_ori_val,
                    'x_mirror': x_mirror, 'x_mirror_256': x_mirror_256, 'c_mirror': c_mirror,
                    'depth_mirror': depth_mirror,
                }
                batch_inp = self.forward(batch)
            


            # Logging related

            y_hat, y_hat_novel = batch_inp['y_hat'], batch_inp['y_hat_novel']
            y_hat_resized, y_hat_novel_resized = batch_inp['y_hat_resized'], batch_inp['y_hat_novel_resized']
            warp_img, warp_warp_img = batch_inp['warp_img'].cpu().detach(), batch_inp['warp_warp_img'].cpu().detach()
            mask, mask_inv = batch_inp['mask'].cpu().detach(), batch_inp['mask_inv'].cpu().detach()
            pred_novel, pred_inv_warp = batch_inp['pred_novel'].cpu().detach(), batch_inp['pred_inv_warp'].cpu().detach()
            
            _x = (x if self.opts.warp.warp_ori_val else x_256).cpu().detach()
            _x_mirror = (x_mirror if self.opts.warp.warp_ori_val else x_mirror_256).cpu().detach()
            _y_hat = (y_hat if self.opts.warp.warp_ori_val else y_hat_resized).cpu().detach()
            _y_hat_novel = (y_hat_novel if self.opts.warp.warp_ori_val else y_hat_novel_resized).cpu().detach()

            fake_novel = (warp_img * (1 - mask) + mask * _y_hat_novel).cpu().detach()
            fake_inv_warp = (warp_warp_img * (1 - mask_inv) + mask_inv * _y_hat).cpu().detach()


            res_x = torch.cat([_x[0], _x_mirror[0]], dim=1)
            res_y = torch.cat([_y_hat_novel[0], _y_hat[0]], dim=1)
            res_warp = torch.cat([warp_img[0], warp_warp_img[0]], dim=1)
            res_mask = torch.cat([mask[0], mask_inv[0]], dim=1).expand(3, -1, -1)
            res_fake = torch.cat([fake_novel[0], fake_inv_warp[0]], dim=1)
            res_pred = torch.cat([pred_novel[0], pred_inv_warp[0]], dim=1)

            if self.opts.generator.input_mirror:
                warp_img_mirror = batch_inp['warp_img_mirror'].cpu().detach()
                warp_warp_img_mirror = batch_inp['warp_warp_img_mirror'].cpu().detach()
                mask_mirror = batch_inp['mask_mirror'].cpu().detach()
                mask_inv_mirror = batch_inp['mask_inv_mirror'].cpu().detach()

                fake_novel_mirror = (warp_img_mirror * (1 - mask_mirror) + mask_mirror * _y_hat_novel).cpu().detach()
                fake_inv_warp_mirror = (warp_warp_img_mirror * (1 - mask_inv_mirror) + mask_inv_mirror * _y_hat).cpu().detach()

                res_warp_mirror = torch.cat([warp_img_mirror[0], warp_warp_img_mirror[0]], dim=1)
                res_mask_mirror = torch.cat([mask_mirror[0], mask_inv_mirror[0]], dim=1).expand(3, -1, -1)
                res_fake_mirror = torch.cat([fake_novel_mirror[0], fake_inv_warp_mirror[0]], dim=1)
                res = torch.cat([res_x, res_y, res_warp, res_mask, res_fake, res_warp_mirror, res_mask_mirror, res_fake_mirror, res_pred], dim=2)
            else:
                res = torch.cat([res_x, res_y, res_warp, res_mask, res_fake, res_pred], dim=2)

            # res = add_grid_lines(res, 512 if self.opts.warp.warp_ori_val else 256)
            
            _save_val_dir = os.path.join(self.save_val_dir, f'{self.global_step:04d}')
            os.makedirs(_save_val_dir, exist_ok=True)
            save_val_path = os.path.join(_save_val_dir, f'{batch_idx:04d}.jpg')
            # common.tensor2im(res).save(save_val_path)
            cv2.imwrite(save_val_path, cv2.cvtColor(self.tensor2im(res), cv2.COLOR_RGB2BGR))


            save_single_dir = os.path.join(_save_val_dir, 'single')
            os.makedirs(save_single_dir, exist_ok=True)

            depth, depth_novel = batch_inp['depth'].cpu().detach(), batch_inp['depth_novel'].cpu().detach()
            depth = F.interpolate(depth, size=(512, 512), mode='bicubic', align_corners=False)
            depth_novel = F.interpolate(depth_novel, size=(512, 512), mode='bicubic', align_corners=False)
            # res_depth = torch.cat([depth_novel[0], depth[0]], dim=1).repeat(3, 1, 1)
            save_val_depth_path = os.path.join(save_single_dir, f'{batch_idx:04d}_depth.jpg')
            cv2.imwrite(save_val_depth_path, cv2.cvtColor(self.tensor2im(depth[0].repeat(3, 1, 1), norm=True), cv2.COLOR_RGB2BGR))
            save_val_depth_novel_path = os.path.join(save_single_dir, f'{batch_idx:04d}_depth_novel.jpg')
            cv2.imwrite(save_val_depth_novel_path, cv2.cvtColor(self.tensor2im(depth_novel[0].repeat(3, 1, 1), norm=True), cv2.COLOR_RGB2BGR))

            save_single_list = [_x[0], _x_mirror[0], _y_hat_novel[0], _y_hat[0], warp_img[0], warp_warp_img[0], mask[0].expand(3, -1, -1), mask_inv[0].expand(3, -1, -1), fake_novel[0], fake_inv_warp[0], pred_novel[0], pred_inv_warp[0]]
            if self.opts.generator.input_mirror:
                save_single_list += [warp_img_mirror[0], warp_warp_img_mirror[0], mask_mirror[0].expand(3, -1, -1), mask_inv_mirror[0].expand(3, -1, -1), fake_novel_mirror[0], fake_inv_warp_mirror[0], pred_novel[0], pred_inv_warp[0]]
            for i, img in enumerate(save_single_list):
                save_single_path = os.path.join(save_single_dir, f'{batch_idx:04d}_{i}.jpg')
                cv2.imwrite(save_single_path, cv2.cvtColor(self.tensor2im(img), cv2.COLOR_RGB2BGR))

            if batch_idx >= val_img_log_num - 1:
                break



    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.log.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n\n')

    def make_optimizer(self, parameters, kind='adam', **kwargs):
        if kind == 'adam':
            optimizer_class = torch.optim.Adam
        elif kind == 'adamw':
            optimizer_class = torch.optim.AdamW
        elif kind == 'ranger':
            optimizer_class = Ranger
        else:
            raise ValueError(f'Unknown optimizer kind {kind}')
        return optimizer_class(parameters, **kwargs)
        
    def configure_inpaintor_optimizers(self):
        params = list(self.inpaintor.parameters())
        optimizer = self.make_optimizer(params, **self.opts.optimizers.generator)
        return optimizer

    def configure_discriminator_optimizers(self):
        params = list(self.discriminator.parameters())
        optimizer = self.make_optimizer(params, **self.opts.optimizers.discriminator)
        return optimizer

    def configure_datasets(self):
        
        load_conf_map = True if self.opts.losses.mirror.weight > 1e-5 else False

        self.train_dataset = ImageFolderDataset(path=dataset_paths['train'],
                                                resolution=None,
                                                use_labels=True,
                                                load_conf_map=load_conf_map,
                                                datast_json=self.dataset_json,)
        self.test_dataset  = ImageFolderDataset(path=dataset_paths['test'],
                                                resolution=None,
                                                use_labels=True,
                                                load_conf_map=load_conf_map,
                                                max_size=1000)
        self.train_synth_dataset = SynthImageFolderDataset(path_synth=dataset_paths['synth'],
                                                           total_imgs_num=100000,
                                                           resolution=None,)
        
        print(f"Loading train dataset from {dataset_paths['train']}. Number of training samples: {len(self.train_dataset)}")
        print(f"Loading train synth dataset from {dataset_paths['synth']}. Number of Synth training samples: {len(self.train_synth_dataset)}")
        print(f"Loading test dataset from {dataset_paths['test']}. Number of test samples: {len(self.test_dataset)}")

    def configure_dataloaders(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=int(self.opts.data.batch_size),
                                           shuffle=True,
                                           num_workers=int(self.opts.data.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=int(self.opts.data.test_batch_size),
                                          shuffle=True,  # False
                                          num_workers=int(self.opts.data.test_workers),
                                          drop_last=True)
        self.train_synth_dataloader = DataLoader(self.train_synth_dataset,
                                                 batch_size=int(self.opts.data.batch_size),
                                                 shuffle=True,
                                                 num_workers=int(self.opts.data.workers),
                                                 drop_last=True)

    def cal_inpaintor_loss(self, batch):
        img = batch['image']
        pred_inv_warp, pred_inv_pred = batch['pred_inv_warp'], batch['pred_inv_pred']
        mask_inv = batch['mask_inv']
        codes = batch['codes'] 

        if pred_inv_pred is None:
            batch_rec = {
                'image': img,
                'pred' : pred_inv_warp,
                'mask' : mask_inv,
                'codes': codes,
            }
        else:
            batch_rec = {
                'image': img.repeat(2, 1, 1, 1),
                'pred' : torch.cat([pred_inv_warp, pred_inv_pred], dim=0),
                'mask' : mask_inv.repeat(2, 1, 1, 1),
                'codes': torch.cat([codes, codes], dim=0),
            }
        total_loss, metrics = self.cal_inpaintor_rec_loss(batch_rec, conf_map=None)


        # latent code loss
        if self.opts.losses.latent.weight > 1e-5:
            pred_novel_256 = F.adaptive_avg_pool2d(batch['pred_novel'], (256, 256))
            w_pred_novel = self.gan.encoder_forward(pred_novel_256)
            latent_value = F.mse_loss(codes, w_pred_novel)

            total_loss = total_loss + latent_value * self.opts.losses.latent.weight
            metrics['gen_latent'] += latent_value
        
        metrics['gen_loss'] = total_loss
        metrics = {k: float(v) for k, v in metrics.items()}
        
        return total_loss, metrics

    def cal_inpaintor_mirror_loss(self, batch, conf_map=None, rec_only=True):

        batch_rec = {
            'image': batch['image_mirror'],
            'pred' : batch['pred_mirror'],
            'mask' : batch['mask'],
            'codes': batch['codes'],
        }
        loss_rec, metrics_rec = self.cal_inpaintor_rec_loss(batch_rec, conf_map=conf_map)

        if rec_only:
            total_loss = loss_rec
            metrics = {**metrics_rec}

        else:
            loss_ori, metrics_ori = self.cal_inpaintor_loss(batch)
            
            total_loss = loss_rec + loss_ori
            metrics = {**metrics_rec, **metrics_ori}

        metrics['gen_loss'] = total_loss
        metrics = add_suffix_to_keys(metrics, '_mirror')
        metrics = {k: float(v) for k, v in metrics.items()}

        return total_loss, metrics
        

    def cal_inpaintor_rec_loss(self, batch, conf_map):

        img = batch['image']
        predicted_img = batch['pred']
        mask = batch['mask'] if self.opts.losses.with_mask else torch.zeros_like(batch['mask'])
        codes = batch['codes']

        if conf_map is not None:
            _, _, h, w = img.size()
            conf_map = F.interpolate(conf_map.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)

        metrics = {}
        total_loss = 0

        # L1
        if self.opts.losses.l1.weight_known > 1e-5:
            l1_value = masked_l1_loss(pred=predicted_img, target=img, mask=mask if conf_map is None else mask / (conf_map + EPS),
                                      weight_known=self.opts.losses.l1.weight_known,
                                      weight_missing=self.opts.losses.l1.weight_missing)

            total_loss = total_loss + l1_value
            metrics['gen_l1'] = l1_value
        
        # MSE
        if self.opts.losses.mse.weight_known > 1e-5:
            mse_value = masked_l2_loss(pred=predicted_img, target=img, mask=mask if conf_map is None else mask / (conf_map + EPS),
                                       weight_known=self.opts.losses.mse.weight_known,
                                       weight_missing=self.opts.losses.mse.weight_missing)

            total_loss = total_loss + mse_value
            metrics['gen_mse'] = mse_value

        # vgg-based perceptual loss
        if self.opts.losses.perceptual.weight > 1e-5:
            pl_value = self.loss_pl(predicted_img, img, mask=mask).sum()

            total_loss = total_loss + pl_value * self.opts.losses.perceptual.weight
            metrics['gen_pl'] = pl_value
        
        # LPIPS
        if self.opts.losses.lpips.weight > 1e-5:
            lpips_value = self.loss_lpips(predicted_img, img)  # mask

            total_loss = total_loss + lpips_value * self.opts.losses.lpips.weight
            metrics['gen_lpips'] = lpips_value

        # ID
        if self.opts.losses.id.weight > 1e-5:
            id_value, sim_improvement_value, _ = self.loss_id(predicted_img, img, img)

            total_loss = total_loss + id_value * self.opts.losses.id.weight
            metrics['gen_id'] = id_value
            metrics['gen_id_improve'] = sim_improvement_value

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.inpaintor, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.opts.losses.feature_matching.weight > 1e-5:
            # need_mask_in_fm = OmegaConf.to_container(self.opts.losses.feature_matching).get('pass_mask', False)
            # mask_for_fm = supervised_mask if need_mask_in_fm else None
            mask_for_fm = None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.opts.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        if self.opts.losses.resnet_pl.weight > 1e-5:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value
        
        # latent code loss
        if self.opts.losses.latent.weight > 1e-5:
            pred_256 = F.adaptive_avg_pool2d(batch['pred'], (256, 256))
            w_pred_novel = self.gan.encoder_forward(pred_256)
            latent_value = F.mse_loss(codes, w_pred_novel)

            total_loss = total_loss + latent_value * self.opts.losses.latent.weight
            metrics['gen_latent'] = latent_value

        return total_loss, metrics

    def cal_discriminator_loss(self, batch):

        total_loss = 0
        metrics = {}

        predicted_img = batch['pred'].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.inpaintor, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        metrics['discr_loss'] = total_loss
        metrics = {k: float(v) for k, v in metrics.items()}

        return total_loss, metrics

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.log.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(get_time(), f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, y_hat, y_hat_novel, warp_img, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'y_hat': common.tensor2im(y_hat[i]),
                'y_hat_novel': common.tensor2im(y_hat_novel[i]),
                'warp': common.tensor2im(warp_img[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            # '3dgan_state_dict': self.gan.state_dict(),
            'inpaintor_state_dict': self.inpaintor.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opts': vars(self.opts),
            'best_val_loss': self.best_val_loss,
            'step': self.global_step,
            'inpaintor_optimizer': self.optimizer_inpaintor.state_dict(),
            'discriminator_optimizer': self.optimizer_discriminator.state_dict()
        }
        # if self.opts.use_discriminator:
        #     save_dict['dis_state_dict'] = self.discriminator.state_dict()
        #     save_dict['dis_optimizer'] = self.optimizer_discriminator.state_dict()
        return save_dict

    def resume(self):
        print("Resume from", self.opts.checkpoint_path)
        ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
        
        if "step" in ckpt:
            self.global_step = ckpt["step"]
            print(f"Resuming training process from step {self.global_step}")

        # if '3dgan_state_dict' in ckpt:
        #     self.gan.load_state_dict(ckpt['state_dict'])
        #     print(f"Resuming WplusNet from step {self.global_step}")

        if "discriminator_state_dict" in ckpt:
            self.discriminator.load_state_dict(ckpt["discriminator_state_dict"], strict=True)
            print(f"Resuming Discriminator from step {self.global_step}")
        
        if "inpaintor_state_dict" in ckpt:
            self.inpaintor.load_state_dict(ckpt["inpaintor_state_dict"], strict=True)
            print(f"Resuming Inpaintor from step {self.global_step}")

        if "inpaintor_optimizer" in ckpt:
            self.optimizer_inpaintor.load_state_dict(ckpt["inpaintor_optimizer"])
            print("Load inpaintor optimizer from checkpoint")

        if "discriminator_optimizer" in ckpt:
            self.optimizer_discriminator.load_state_dict(ckpt["discriminator_optimizer"])
            print("Load Discriminator optimizer from checkpoint")

        if "best_val_loss" in ckpt:
            self.best_val_loss = ckpt["best_val_loss"]
            print(f"Current best val loss: {self.best_val_loss }")

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

    # def get_ffhq_camera_params(self):
    #     with open(os.path.join(dataset_paths['train'], self.dataset_json), 'r') as f:
    #         ffhq_cam_list = json.load(f)['labels']
        
    #     ffhq_cam_list = [[x[1]] for x in ffhq_cam_list]
    #     return ffhq_cam_list

    def tensor2im(self, var, norm=False):
        var = var.cpu().detach().permute(1, 2, 0).numpy()
        if norm:
            var = (var - var.min()) / (var.max() - var.min()) * 255
        else:
            # var = (var + 1) / 2
            var = np.clip(var, 0, 1)
            var = var * 255
        return var.astype('uint8')
    
    def get_mirror_c(self, c):
        pose, intrinsics = c[:, :16].reshape(-1, 4, 4), c[:, 16:].reshape(-1, 3, 3)
        flipped_pose = self.flip_yaw(pose)
        c_mirror = torch.cat([flipped_pose.reshape(-1, 4 * 4), intrinsics.reshape(-1, 3 * 3)], dim=1).reshape(-1, 25)
        return c_mirror

    def flip_yaw(self, pose_matrix):
        flipped = pose_matrix.clone()
        flipped[:, 0, 1] *= -1
        flipped[:, 0, 2] *= -1
        flipped[:, 1, 0] *= -1
        flipped[:, 2, 0] *= -1
        flipped[:, 0, 3] *= -1
        return flipped