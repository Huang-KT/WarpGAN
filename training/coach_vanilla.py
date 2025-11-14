import os
import json
import random
import numpy as np
from PIL import Image
import copy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from datasets.dataset_vanilla import ImageFolderDataset
from configs.paths_config import dataset_paths
from configs.paths_config import model_paths

from models.wplusnet import WplusNet
from models.stylegan2.stylegan_ada import Discriminator
from models.eg3d.camera_utils import LookAtPoseSampler

from training.ranger import Ranger
from criteria import id_loss, moco_loss
from criteria.lpips.lpips import LPIPS
from criteria.w_norm import WNormLoss
from utils import common, train_utils
from utils.depth_utils import calibrate_disparity


EPS = 1


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0
        self.device = self.opts.device

        if self.opts.log.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        if self.opts.gan.eg3d_type is None:
            t = 'plus' if self.opts.data.add_LPFF else 'ori'
            t += '_r' if self.opts.data.rebalanced else ''
            self.opts.gan.eg3d_type = t

        # Initialize network
        self.net = WplusNet(self.opts.gan).to(self.device)

        if self.opts.disc.able:
            d_init_args = {
                'c_dim': 0, 'img_resolution': 512, 'img_channels': 3, 'architecture': 'resnet', 
                'channel_base': 32768, 'channel_max': 512, 'num_fp16_res': 4, 'conv_clamp': 256, 'cmap_dim': None,
                'block_kwargs': {
                    'activation': 'lrelu', 'resample_filter': [1, 3, 3, 1], 'freeze_layers': 0
                },
                'mapping_kwargs': {
                    'num_layers': 0, 'embed_features': None, 'layer_features': None,
                    'activation': 'lrelu', 'lr_multiplier': 0.1
                },
                'epilogue_kwargs': {
                    'mbstd_group_size': None, 'mbstd_num_channels': 1, 'activation': 'lrelu'
                }
            }
            self.discriminator = Discriminator(**d_init_args).to(self.opts.device).float()
            if self.opts.disc.checkpoint_path is None:
                print("Training discriminator from scratch")
            else:
                self.discriminator.load_state_dict(torch.load(self.opts.disc.checkpoint_path), strict=False)
                print(f"Loading discriminator checkpoint from {self.opts.disc.checkpoint_path}")
        
        
        if self.opts.data.add_LPFF:
            self.cam_json = 'dataset_plus_rebalanced.json'
        else:
            self.cam_json = 'dataset_rebalanced.json'
        print("Camera json: ", self.cam_json)
        
        self.ffhq_cam_list = self.get_ffhq_camera_params()


        # Initialize losses
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.losses.real.ori.lpips > 1e-5:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.losses.real.ori.id > 1e-5:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.losses.real.ori.w_norm > 0:
            self.w_norm_loss = WNormLoss().to(self.device).eval()


        # Initialize optimizer
        # self.optimizer_triplane = self.configure_triplane_optimizers()
        self.optimizer_psp = self.configure_psp_optimizers()
        if self.opts.disc.able:
            self.optimizer_discriminator = self.configure_discriminator_optimizers()


        # Set requires_grad to False
        self.net.psp_encoder.requires_grad_(False)
        if self.opts.disc.able:
            self.discriminator.requires_grad_(False)


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
        
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.configure_dataloaders()


        # Initialize logger
        log_dir = os.path.join(self.opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)


        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(self.opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.log.save_interval is None:
            self.opts.log.save_interval = self.opts.max_steps


        # Resume training process from checkpoint path
        if self.opts.checkpoint_path is not None:
            self.resume()


        # Make dirs for saving images
        self.save_train_dir = os.path.join(self.logger.log_dir, 'images/train')
        self.save_val_dir = os.path.join(self.logger.log_dir, 'images/val')
        os.makedirs(self.save_train_dir, exist_ok=False)
        os.makedirs(self.save_val_dir, exist_ok=False)


    def train(self):
        self.net.train()
        if self.opts.disc.able:
            self.discriminator.train()
        # torch.cuda.empty_cache()

        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):

                # x_256, x, c, x_mirror_256, x_mirror, c_mirror, conf_map_mirror, depth_map_est, _ = batch
                x_256, x, c = batch[:3]
                x_256, x, c = x_256.to(self.device).float(), x.to(self.device).float(), c.to(self.device).float()

                if self.opts.losses.real.mirror.weight > 1e-5:
                    x_mirror_256, x_mirror, c_mirror, conf_map_mirror = batch[3:7]
                    x_mirror_256, x_mirror = x_mirror_256.to(self.device).float(), x_mirror.to(self.device).float()
                    c_mirror = c_mirror.to(self.device).float()
                    conf_map_mirror = conf_map_mirror.to(self.device).float()
                # depth_map_est = depth_map_est.to(self.device).float().clone().detach() if self.opts.losses.real.ori.depth > 1e-5 else None


                loss_dict = {}



                # =============================== Update G from Real Image ===============================

                self.net.psp_encoder.requires_grad_(True)

                loss_real = 0.0
                
                if self.opts.losses.real.mirror.weight > 1e-5:
                    ###################################
                    outs = self.net(x_256, c, c_mirror)
                    ###################################
                    sampled_c = c_mirror.clone()
                    loss_psp, loss_dict, id_logs = self.calc_loss_psp(x_256, outs["y_hat_resized"], outs["codes"], loss_dict, outs["depth"])
                    loss_real += loss_psp
                    if self.opts.losses.real.mirror.weight > 1e-5:
                        loss_psp_mirror, loss_dict, _ = self.calc_mirror_loss_psp(x_256.clone(), x_mirror_256, outs["y_hat_novel_resized"], conf_map_mirror, loss_dict, outs["depth_novel"])
                        loss_real += loss_psp_mirror * self.opts.losses.real.mirror.weight
                else:
                    if (self.opts.disc.able and self.global_step >= self.opts.disc.add_disc):
                        sampled_c = self.sample_camera_poses(x.size(0))
                        ####################################
                        outs = self.net(x_256, c, sampled_c)
                        ####################################
                    else:
                        #########################
                        outs = self.net(x_256, c)
                        #########################
                    loss_psp, loss_dict, id_logs = self.calc_loss_psp(x_256, outs["y_hat_resized"], outs["codes"], loss_dict, outs["depth"])
                    loss_real += loss_psp
                    

                # self.optimizer_psp.zero_grad()
                # loss_psp.backward()
                # self.optimizer_psp.step()
                # self.optimizer_psp.zero_grad()


                # Disc Adv Loss
                if self.opts.disc.able and self.global_step >= self.opts.disc.add_disc:
                    # sampled_c = self.sample_camera_poses(x.size(0))
                    # #################################################
                    # outs_disc = self.net(x_256.clone(), c, sampled_c)
                    # #################################################
                    # sampled_c = c_mirror.clone()
                    # outs_disc = {k: v.clone() for k, v in outs.items()}

                    assert self.opts.losses.disc.adv > 1e-5

                    y_hat_disc, depth_disc = outs["y_hat"], outs["depth"]
                    y_hat_novel_disc, depth_novel_disc = outs["y_hat_novel"], outs["depth_novel"]

                    # input_adv_disc = y_hat_disc if random.random() > 0.5 else y_hat_novel_disc
                    input_adv_disc = y_hat_novel_disc

                    fake_preds = self.discriminator({'image': input_adv_disc}, c=None)
                    loss_G_adv = self.g_nonsaturating_loss(fake_preds)
                    loss_dict["loss_G_adv"] = float(loss_G_adv)

                    loss_real += loss_G_adv * self.opts.losses.disc.adv

                    # self.optimizer_psp.zero_grad()
                    # loss_G_adv.backward()
                    # self.optimizer_psp.step()
                    # self.optimizer_psp.zero_grad()


                loss_dict['loss_real'] = float(loss_real)
                
                self.optimizer_psp.zero_grad()
                loss_real.backward()
                # nn.utils.clip_grad_norm_(self.net.psp_encoder.parameters(), max_norm=10.0, norm_type=2)
                self.optimizer_psp.step()
                self.optimizer_psp.zero_grad()


                self.net.psp_encoder.requires_grad_(False)




                # =============================== Update D from Real Image ===============================

                # Update Disc
                if self.opts.disc.able and self.global_step >= self.opts.disc.add_disc:

                    self.discriminator.requires_grad_(True)
                    
                    d_loss, loss_dict = self.calc_discriminator_loss(loss_dict, {'image': input_adv_disc.clone().detach()}, {'image': x.clone().detach()}, r_s='real')

                    # R1 Regularization
                    if self.global_step % self.opts.losses.disc.d_reg_every == 0:
                        d_r1_loss, loss_dict = self.calc_discriminator_r1_loss(loss_dict, {'image': x.clone().detach()})

                        d_loss += d_r1_loss[0]
                        loss_dict["loss_D_real"] = float(d_loss)
                    
                    self.optimizer_discriminator.zero_grad()
                    d_loss.backward()
                    # nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0, norm_type=2)
                    self.optimizer_discriminator.step()
                    self.optimizer_discriminator.zero_grad()

                    self.discriminator.requires_grad_(False)



                # =============================== Logging Related ===============================

                if self.global_step % self.opts.log.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):

                    _blank = torch.zeros_like(x[0])

                    if self.opts.losses.real.mirror.weight > 1e-5:
                        y_hat, y_hat_novel = outs['y_hat'], outs['y_hat_novel']
                        res = torch.cat([x[0], y_hat[0], y_hat_novel[0]], dim=2).cpu().detach()
                    else:
                        res = torch.cat([x[0], outs['y_hat'][0]], dim=2).cpu().detach()


                    res = common.add_grid_lines(res, 512)

                    save_train_path = os.path.join(self.save_train_dir, f'{self.global_step:04d}.jpg')
                    cv2.imwrite(save_train_path, cv2.cvtColor(self.tensor2im(res), cv2.COLOR_RGB2BGR))

                    # self.parse_and_log_images(id_logs, x, final_out, final_out_novel, title='images/train')


                if self.global_step % self.opts.log.board_interval == 0 or (self.global_step < 100 and self.global_step % 25 == 0):
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Log images of first batch to wandb
                # if self.opts.log.use_wandb and batch_idx == 0 and self.phase == "first":
                #     self.wb_logger.log_images_to_wandb(x, final_out, final_out_novel, id_logs, prefix="train", step=self.global_step, opts=self.opts)
                

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.log.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict is not None:
                        if 'loss_triplane' in val_loss_dict:
                            new_val_loss = val_loss_dict['loss_triplane']
                        else:
                            new_val_loss = val_loss_dict['loss_psp']

                    if val_loss_dict and (self.best_val_loss is None or new_val_loss < self.best_val_loss):
                        self.best_val_loss = new_val_loss
                        self.checkpoint_me(val_loss_dict, is_best=True)


                if self.global_step > 0 and (self.global_step % self.opts.log.save_interval == 0 or self.global_step == self.opts.max_steps):
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    # else:
                    #     self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break
                
                ######################################################
                # if self.opts.gan.encoder_type == 'goae':
                #     self.net.psp_encoder.set_stage(self.global_step)
                ######################################################

                self.global_step += 1

    def validate(self):
        self.net.eval()
        # torch.cuda.empty_cache()

        agg_loss_dict = []
        val_img_log_num = 100
        for batch_idx, batch in enumerate(self.test_dataloader):

            x_256, x, c = batch[:3]
            x_256, x, c = x_256.to(self.device).float(), x.to(self.device).float(), c.to(self.device).float()
            if self.opts.losses.real.mirror.weight > 1e-5:
                x_mirror_256, x_mirror, c_mirror, conf_map_mirror = batch[3:7]
                x_mirror_256, x_mirror, c_mirror = x_mirror_256.to(self.device).float(), x_mirror.to(self.device).float(), c_mirror.to(self.device).float()
                conf_map_mirror = conf_map_mirror.to(self.device).float()
            # if self.opts.losses.real.ori.depth > 1e-5:
            #     depth_map_est = depth_map_est.to(self.device).float()


            cur_loss_dict = {}
            
            if self.opts.losses.real.mirror.weight > 1e-5:
                c_novel = c_mirror.clone()
                with torch.no_grad():
                    ###################################
                    outs = self.net(x_256, c, c_mirror)
                    ###################################
                _, cur_loss_dict, id_logs = self.calc_loss_psp(x_256, outs["y_hat_resized"], outs["codes"], cur_loss_dict, outs["depth"])
                _, cur_loss_dict, _ = self.calc_mirror_loss_psp(x_256, x_mirror_256, outs["y_hat_novel_resized"], conf_map_mirror, cur_loss_dict, outs["depth_novel"])
            else:
                c_novel = self.sample_camera_poses(x.size(0))
                with torch.no_grad():
                    ###################################
                    outs = self.net(x_256, c, c_novel)
                    ###################################
                _, cur_loss_dict, id_logs = self.calc_loss_psp(x_256, outs["y_hat_resized"], outs["codes"], cur_loss_dict, outs["depth"])
                
            agg_loss_dict.append(cur_loss_dict)
            

            # Logging related
            if batch_idx < val_img_log_num:

                res = torch.cat([x[0], outs["y_hat"][0], outs["y_hat_novel"][0]], dim=2)    

                res = common.add_grid_lines(res, 512)

                _save_val_dir = os.path.join(self.save_val_dir, f'{self.global_step:04d}')
                os.makedirs(_save_val_dir, exist_ok=True)
                save_val_path = os.path.join(_save_val_dir, f'{batch_idx:04d}.jpg')
                cv2.imwrite(save_val_path, cv2.cvtColor(self.tensor2im(res), cv2.COLOR_RGB2BGR))
                


            # Log images of first batch to wandb
            # if self.opts.log.use_wandb and batch_idx == 0:
            #     self.wb_logger.log_images_to_wandb(x, final_out, final_out_novel, id_logs, prefix="test", step=self.global_step, opts=self.opts)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 20:  # and not self.compare:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

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
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

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

    def configure_psp_optimizers(self):
        params = list(self.net.psp_encoder.parameters())
        optimizer = self.make_optimizer(params, **self.opts.optimizers.psp)
        return optimizer

    def configure_discriminator_optimizers(self):
        params = list(self.discriminator.parameters())
        optimizer = self.make_optimizer(params, **self.opts.optimizers.disc)
        return optimizer

    def configure_datasets(self):
        
        load_depth = True if self.opts.losses.real.ori.depth > 1e-5 else False
        load_conf_map = True if self.opts.losses.real.mirror.weight > 1e-5 else False
        use_mirror = True if self.opts.losses.real.mirror.weight > 1e-5 else False
        
        train_dataset = ImageFolderDataset(path=dataset_paths['train'],
                                           resolution=None, 
                                           load_conf_map=load_conf_map,
                                           use_mirror=use_mirror,
                                           use_labels=True,
                                           load_depth=load_depth,
                                           datast_json=self.dataset_json)
        test_dataset = ImageFolderDataset(path=dataset_paths['test'],
                                          resolution=None, 
                                          load_conf_map=load_conf_map,
                                          use_mirror=use_mirror,
                                          use_labels=True,
                                          load_depth=load_depth,
                                          max_size=1000)
        
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def configure_dataloaders(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.data.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.data.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.data.test_batch_size,
                                          shuffle=True,
                                          num_workers=int(self.opts.data.test_workers),
                                          drop_last=True)

    def calc_loss_psp(self, x, y_hat, w, loss_dict, depth_map=None, depth_map_est=None):
        loss = 0.0
        id_logs = None
        
        if self.opts.losses.real.ori.l2 > 1e-5:
            loss_l2 = F.mse_loss(y_hat, x)
            loss_dict['loss_l2_psp'] = float(loss_l2)
            loss += loss_l2 * self.opts.losses.real.ori.l2

        if self.opts.losses.real.ori.lpips > 1e-5:
            loss_lpips = self.lpips_loss(y_hat, x)
            loss_dict['loss_lpips_psp'] = float(loss_lpips)
            loss += loss_lpips * self.opts.losses.real.ori.lpips

        if self.opts.losses.real.ori.id > 1e-5:
            loss_id, sim_improvement, _ = self.id_loss(y_hat, x, x)
            loss_dict['loss_id_psp'] = float(loss_id)
            loss_dict['id_improve_psp'] = float(sim_improvement)
            loss += loss_id * self.opts.losses.real.ori.id
        
        if self.opts.losses.real.ori.depth > 1e-5:
            assert depth_map is not None and depth_map_est is not None
            depth_reg = self.depth_reg_loss(depth_map, depth_map_est, align=True)
            loss_dict["loss_depth_psp"] = float(depth_reg)
            loss += depth_reg * self.opts.losses.real.ori.depth

        if self.opts.losses.real.ori.w_norm > 1e-5:
            loss_w_norm = self.w_norm_loss(w, self.net.latent_avg)
            loss_dict['loss_w_norm_psp'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.losses.real.ori.w_norm

        if self.opts.gan.encoder_type == 'goae' and self.opts.losses.real.ori.w_delta_reg > 1e-5:
            w_delta = w - w[:,[0],:]
            loss_w_delta_reg = torch.sum(w_delta.norm(2, dim=(1, 2))) / w_delta.shape[0]
            loss_dict["loss_w_delta_reg_psp"] = float(loss_w_delta_reg)
            loss += loss_w_delta_reg * self.opts.losses.real.ori.w_delta_reg
        
        loss_dict['loss_psp'] = float(loss)
        return loss, loss_dict, id_logs

    def calc_mirror_loss_psp(self, x, x_mirror, y_hat, conf_map, loss_dict, depth_map=None, depth_map_est=None):
        loss = 0.0
        id_logs = None
        
        if self.opts.losses.real.mirror.l2 > 1e-5:
            if self.opts.losses.real.mirror.normalize_mirror:
                loss_l2 = torch.square(x_mirror - y_hat)
                loss_l2 = loss_l2.mean(dim=1)
                loss_l2 = loss_l2 / (conf_map + EPS)
                loss_l2 = loss_l2.mean(dim=(1, 2))
                conf_map_mean = (1 / ( conf_map + EPS)).mean(dim=(1, 2))
                loss_l2 = (loss_l2 / conf_map_mean).mean()
            else:
                loss_l2 = torch.square(x_mirror - y_hat)
                loss_l2 = loss_l2.mean(dim=1)
                if conf_map is not None:
                    loss_l2 = (loss_l2 / (conf_map + EPS ))
                loss_l2 = loss_l2.mean()
            loss_dict['loss_l2_psp_mirror'] = float(loss_l2)
            loss += loss_l2 * self.opts.losses.real.mirror.l2

        if self.opts.losses.real.mirror.lpips > 1e-5:
            loss_lpips = self.lpips_loss(y_hat, x_mirror)
            loss_dict['loss_lpips_psp_mirror'] = float(loss_lpips)
            loss += loss_lpips * self.opts.losses.real.mirror.lpips
            
        if self.opts.losses.real.mirror.id > 1e-5:
            loss_id, sim_improvement, _ = self.id_loss(y_hat, x_mirror, x_mirror)
            loss_dict['loss_id_psp_mirror'] = float(loss_id)
            loss_dict['id_improve_psp_mirror'] = float(sim_improvement)
            loss += loss_id * self.opts.losses.real.mirror.id

        # if self.opts.losses.real.mirror.depth > 1e-5:
        #     assert depth_map is not None and depth_map_est is not None
        #     depth_map_est_mirror = torch.flip(depth_map_est, dims=[2])
        #     depth_reg = self.depth_reg_loss(depth_map, depth_map_est_mirror, align=True)
        #     loss_dict["loss_depth_psp_mirror"] = float(depth_reg)
        #     loss += depth_reg * self.opts.losses.real.mirror.depth
        
        loss_dict['loss_psp_mirror'] = float(loss)
        return loss, loss_dict, id_logs
    
    def depth_reg_loss(self, depth_map, depth_map_est, align=False):
        if align:  # align depth
            depth_map_est = F.interpolate(depth_map_est.unsqueeze(1), size=(depth_map.shape[2], depth_map.shape[3]), mode='bilinear', align_corners=True)
            depth_map_aligned = calibrate_disparity(depth_map_est.squeeze(), depth_map.squeeze())
        else:
            depth_map_aligned = depth_map_est
        
        # Charbonnier penalty for output depth `depth_map_aligned` and off-the-shelf depth `depth_map`
        depth_reg = torch.mean(torch.sqrt(torch.pow(depth_map_aligned.squeeze() - depth_map.squeeze(), 2) + 1e-6))
        return depth_reg

    def calc_latent_loss(self, ws, ws_inv, loss_dict):
        loss = 0.0
        
        # mse
        if self.opts.losses.synth.latent.l2 > 1e-5:
            loss_latent_l2 = F.mse_loss(ws, ws_inv)  # ws.repeat(ws_inv.shape[0], 1, 1)
            loss_dict['loss_l2_latent_synth'] = float(loss_latent_l2)
            loss += loss_latent_l2 * self.opts.losses.synth.latent.l2
        
        # std
        if self.opts.losses.synth.latent.std > 1e-5:
            b = self.opts.data.synth.batch_size
            per_num = self.opts.data.synth.per_num

            loss_latent_std = torch.std(ws_inv.view(b, per_num, 14, self.net.decoder.z_dim), dim=1)
            loss_latent_std = torch.mean(loss_latent_std)
            loss_dict["loss_std_latent_synth"] = float(loss_latent_std)
            loss += loss_latent_std * self.opts.losses.synth.latent.std

        loss_dict['loss_latent_synth'] = float(loss)
        return loss, loss_dict


    def g_nonsaturating_loss(self, fake_preds):
        loss = F.softplus(-fake_preds).mean()
        return loss
    
    def calc_discriminator_loss(self, loss_dict, generated_images, real_images, r_s):
        fake_preds = self.discriminator(generated_images, c=None)
        real_preds = self.discriminator(real_images, c=None)
        loss = self.d_logistic_loss(real_preds, fake_preds)
        loss_dict["loss_D_"+r_s] = float(loss)
        return loss, loss_dict
    
    def d_logistic_loss(self, real_preds, fake_preds):
        real_loss = F.softplus(-real_preds)
        fake_loss = F.softplus(fake_preds)

        return (real_loss.mean() + fake_loss.mean()) / 2

    def d_r1_loss(self, real_pred, real_img):
        (grad_real, ) = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img['image'], create_graph=True)
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty
    
    def calc_discriminator_r1_loss(self, loss_dict, real_images):
        real_img_tmp_image = real_images['image'].detach().requires_grad_(True)
        real_img_tmp = {'image': real_img_tmp_image}

        real_preds = self.discriminator(real_img_tmp, c=None)
        real_preds = real_preds.view(real_img_tmp_image.size(0), -1)
        real_preds = real_preds.mean(dim=1).unsqueeze(1)
        
        r1_loss = self.d_r1_loss(real_preds, real_img_tmp)
        loss_D_R1 = self.opts.losses.disc.d_r1_gamma / 2 * r1_loss * self.opts.losses.disc.d_reg_every + 0 * real_preds[0]
        loss_dict["loss_D_r1_reg"] = float(loss_D_R1)
        return loss_D_R1, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.log.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(common.get_time(), f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, y_hat, y_hat_novel, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'y_hat': common.tensor2im(y_hat[i]),
                'y_hat_novel': common.tensor2im(y_hat_novel[i]),
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
            'opts': self.opts,  # vars(self.opts),
            'step': self.global_step,
            'best_val_loss': self.best_val_loss,

            'psp_optimizer': self.optimizer_psp.state_dict(),
            'psp_state_dict': self.net.psp_encoder.state_dict(),
        }
        if self.opts.disc.able:
            save_dict['disc_state_dict'] = self.discriminator.state_dict()
            save_dict['disc_optimizer'] = self.optimizer_discriminator.state_dict()
        return save_dict

    def resume(self):
        ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
        
        if "step" in ckpt:
            self.global_step = ckpt["step"]
            print(f"Resuming training process from step {self.global_step}")

        if "best_val_loss" in ckpt:
            self.best_val_loss = ckpt["best_val_loss"]
            print(f"Current best val loss: {self.best_val_loss }")

        if "psp_optimizer" in ckpt:
            self.optimizer_psp.load_state_dict(ckpt["psp_optimizer"])
            print("Load psp optimizer from checkpoint")

        if "disc_optimizer" in ckpt and self.opts.disc.able:
            self.optimizer_discriminator.load_state_dict(ckpt["disc_optimizer"])
            print("Load Discriminator optimizer from checkpoint")

        if "psp_state_dict" in ckpt:
            self.net.psp_encoder.load_state_dict(ckpt["psp_state_dict"])
            print(f"Resuming pSp from step {self.global_step}")

        if "disc_state_dict" in ckpt and self.opts.disc.able:
            self.discriminator.load_state_dict(ckpt["disc_state_dict"])
            print(f"Resuming Discriminator from step {self.global_step}")

    def sample_camera_poses(self, N):
        sampled_poses = random.sample(self.ffhq_cam_list, N)
        sampled_poses = torch.from_numpy(np.array(sampled_poses).reshape(N, -1)).to(self.device).float()
        return sampled_poses

    def get_ffhq_camera_params(self):
        with open(os.path.join(dataset_paths['train'], self.cam_json), 'r') as f:
            ffhq_cam_list = json.load(f)['labels']
        
        ffhq_cam_list = [[x[1]] for x in ffhq_cam_list]
        return ffhq_cam_list

    def get_pose(self, cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7):
        
        if yaw is None:
            yaw = np.random.uniform(-yaw_range, yaw_range)
        if pitch is None:
            pitch = np.random.uniform(-pitch_range, pitch_range)

        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=self.device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
        return c

    def get_foreground_mask(self, depth):
        depth_mean = torch.mean(depth)
        depth_zeros = torch.zeros_like(depth_mean)
        depth_ones = torch.ones_like(depth_mean)
        masked_depths = torch.where(depth < depth_mean, depth_ones, depth_zeros)
        return masked_depths

    def tensor2im(self, var):
        var = var.cpu().detach()
        var = var.permute(1, 2, 0).numpy()
        var = (var + 1) / 2
        var = np.clip(var, 0, 1)
        var = (var * 255).astype(np.uint8)
        return var