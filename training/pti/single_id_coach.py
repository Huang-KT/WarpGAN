import os
import PIL.Image
import cv2
import imageio
from tqdm import tqdm
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import kornia.morphology as km

from models.wplusnet import WplusNet
from models.saicinpainting.training.modules import make_generator
from models.saicinpainting.utils import set_requires_grad
from training.pti.base_coach import BaseCoach
from utils.warp.Splatting import Warper
from models.eg3d.camera_utils import LookAtPoseSampler
from utils import common


def tensor2im(var, norm=False):
    var = var.cpu().detach().permute(1, 2, 0).numpy()
    if norm:
        var = (var - var.min()) / (var.max() - var.min()) * 255
    else:
        var = ((var + 1) / 2)
        var = np.clip(var, 0, 1)
        var = var * 255
    return var.astype('uint8')

def get_pose(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7):
    
    if yaw is None:
        yaw = np.random.uniform(-yaw_range, yaw_range)
    if pitch is None:
            pitch = np.random.uniform(-pitch_range, pitch_range)

    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=cam_pivot.device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
    return c

class SingleIDCoach(BaseCoach):

    def __init__(self, opts, run_name, multi_views, use_wandb):
        super().__init__(opts, run_name, multi_views, use_wandb)

        if self.opts.get_w_pivot != 'opt' or self.multi_views:
            self.gan = WplusNet(self.opts.gan).to(self.device)
            self.gan.eval()
            set_requires_grad(self.gan, False)
        
        if self.multi_views:
            self.inpaintor = make_generator(self.opts, **self.opts.generator).to(self.device)
            inpaintor_state = torch.load(self.opts.ckpt_inpaintor, map_location='cpu')['inpaintor_state_dict']
            self.inpaintor.load_state_dict(inpaintor_state, strict=True)
            print(f"Loading inpaintor weights from {self.opts.ckpt_inpaintor}")
            self.inpaintor.eval()
            set_requires_grad(self.inpaintor, False)

            self.warper = Warper()

        # if self.multi_views:
        cam_pivot = torch.Tensor([0., 0., 0.2]).to(self.device)
        pitch_range, yaw_range = 0.25, 0.35
        # pitch_range, yaw_range = 0.2, 0.2
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=self.device)
        num_keyframes = 30
        self.render_poses = []
        for frame_idx in range(num_keyframes):
            cam2world_pose = LookAtPoseSampler.sample(3.14/2        + yaw_range   * np.sin(2 * 3.14 * frame_idx / (num_keyframes)),
                                                    3.14/2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes)),
                                                    cam_pivot, radius=2.7, device=self.device)
            c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            self.render_poses.append(c)

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

        inp, masked_img, mask = self.get_inp(img, inversion, mask)

        # if self.opts.infer_type == 'coarse':
        #     return masked_img * 2 - 1

        if self.opts.generator.input_mirror == 'cat' or self.opts.generator.input_mirror == 'condition':
            img_mirror, mask_mirror = batch_inp['image_mirror'], batch_inp['mask_mirror']
            img_mirror = ((img_mirror + 1) / 2).clamp(0, 1)
            inp_mirror, masked_img_mirror, mask_mirror = self.get_inp(img_mirror, inversion, mask_mirror, cat_inv=self.opts.generator.input_mirror=='condition')
            inp = torch.cat([inp, inp_mirror], dim=1)
        else:
            masked_img_mirror, mask_mirror = None, None

        if self.opts.generator.kind == 'ffc_resnet':
            pred = self.inpaintor(inp)
        elif self.opts.generator.kind == 'ffc_style_resnet':
            pred = self.inpaintor(inp, ws)
        else:
            raise NotImplementedError(f"Generator kind {self.opts.generator.kind} not implemented")

        return pred * 2 - 1, masked_img * 2 - 1, mask


    def train(self):

        use_ball_holder = True

        cnt_skip = 0

        for image_256, image, pose, c_novel, frame in tqdm(self.dataloader, desc="Dataloader"):
            if cnt_skip < self.opts.data.skip_batch:
                cnt_skip += 1
                continue

            image_256, image, pose = image_256.to(self.device).float(), image.to(self.device).float(), pose.to(self.device).float()
            # c_novel = c_novel.to(self.device).float()
            image_name = frame[0]

            yaw = -np.pi/6
            c_novel = get_pose(cam_pivot=torch.tensor([0, 0, 0.2]).to(self.device),
                               intrinsics=torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).to(self.device),
                               yaw=yaw, pitch=0).repeat(image.shape[0], 1)  # b, 25

            self.restart_training()

            w_pivot = None
            print("Way to get w_pivot:", self.opts.get_w_pivot)

            if self.opts.get_w_pivot == 'opt':
                w_pivot = self.calc_inversions(image, pose, image_name, initial_w=None)

                if self.opts.save.save_wplus:
                    with torch.no_grad():
                        y_hat_novel_wplus, _ = self.forward(w_pivot, c_novel, eval=True)
                    
                    save_novel_wplus_dir = os.path.join(self.opts.exp_dir, "novel_wplus")
                    os.makedirs(save_novel_wplus_dir, exist_ok=True)
                    save_novel_wplus_path = os.path.join(save_novel_wplus_dir, frame[0]+'.png')
                    cv2.imwrite(save_novel_wplus_path, cv2.cvtColor(tensor2im(y_hat_novel_wplus[0]), cv2.COLOR_RGB2BGR))

            elif self.opts.get_w_pivot == 'encoder' or self.opts.get_w_pivot == 'hybrid':
                with torch.no_grad():
                    w_init = self.gan.encoder_forward(image_256)

                if self.opts.get_w_pivot == 'encoder':
                    w_pivot = w_init.clone()
                else:
                    w_pivot = self.calc_inversions(image, pose, image_name, initial_w=w_init)
            else:
                assert NotImplementedError, 'Invalid w_pivot method'

            # w_pivot = None
            # if not self.multi_views:
            #   if hyperparameters.use_last_w_pivots:
            #       w_pivot = self.load_inversions(w_path_dir, image_name)

            #   elif not hyperparameters.use_last_w_pivots or w_pivot is None:
            #       w_pivot = self.calc_inversions(image, pose, image_name)
            # else:
            #       w_pivot = self.load_inversions(w_path_dir, image_name)

            log_images_counter = 0

            if self.multi_views:

                if self.opts.get_w_pivot == 'encoder':
                    w_inp = w_pivot.clone()
                else:
                    with torch.no_grad():
                        w_inp = self.gan.encoder_forward(image_256)

                with torch.no_grad():
                    out_inv = self.gan.decoder.synthesis(w_inp, pose, noise_mode='const')

                inv_imgs = []
                warp_imgs, masks, teacher_imgs = [], [], []
                fix_teacher = True
                if fix_teacher:
                    # for novel_pose in tqdm(self.render_poses, desc="Render Multi Poses for " + image_name):
                    for novel_pose in self.render_poses:
                        with torch.no_grad():
                            outs_novel = self.gan.decoder.synthesis(w_inp, novel_pose, noise_mode='const')
                            _image = (image + 1) / 2
                            warp_img, visable_mask, _ = self.warper.forward_warp(img1=_image, depth1=out_inv["image_depth"], c1=pose, c2=novel_pose)
                            warp_img = warp_img * 2 - 1

                            image_mirror = torch.flip(image, dims=[3])
                            _image_mirror = (image_mirror + 1) / 2
                            depth_mirror = torch.flip(out_inv["image_depth"], dims=[3])
                            pose_mirror = self.get_mirror_c(pose)

                            if self.opts.generator.input_mirror:
                                warp_img_mirror, visable_mask_mirror, _ = self.warper.forward_warp(img1=_image_mirror, depth1=depth_mirror, c1=pose_mirror, c2=novel_pose)
                                warp_img_mirror = warp_img_mirror * 2 - 1
                            else:
                                warp_img_mirror, visable_mask_mirror = None, torch.ones_like(visable_mask)

                            batch_inp = {
                                'image': warp_img, 'inversion': outs_novel['image'],
                                'mask': 1 - visable_mask, 'ws': w_inp,
                                'image_mirror': warp_img_mirror, 'mask_mirror': 1 - visable_mask_mirror,
                            }
                            pred, _, mask = self.inpaint(batch_inp)

                        inv_imgs.append(outs_novel['image'].detach())
                        warp_imgs.append(warp_img.detach())
                        masks.append(mask.detach() * 2 - 1)
                        teacher_imgs.append(pred.detach())

                else:
                    assert NotImplementedError, 'Not implemented'


            if self.multi_views:
               max_pti_steps = self.opts.hyperparameters.max_pti_steps_multiviews
            else:
               max_pti_steps = self.opts.hyperparameters.max_pti_steps
            

            for i in tqdm(range(1, max_pti_steps + 1), desc="Run PTI for " + image_name):
            # for i in range(1, max_pti_steps + 1):
            
                generated_images, _ = self.forward(w_pivot, pose, eval=False)

                loss, l2_loss_val, loss_lpips= self.calc_loss(generated_images, image,
                                                              image_name, self.G, use_ball_holder, w_pivot)

                if self.multi_views:
                    random_pose_id = np.random.choice(np.arange(30), size=1)
                    render_image, _ = self.forward(w_pivot, self.render_poses[random_pose_id[0]], eval=False)
                    
                    teacher_loss, teacher_l2, teacher_lpips = self.calc_loss(render_image, teacher_imgs[random_pose_id[0]],
                                                                             image_name, self.G, use_ball_holder, w_pivot)
                    loss += teacher_loss


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()



                if (loss_lpips <= self.opts.hyperparameters.LPIPS_value_threshold and not self.multi_views) or (i % max_pti_steps == 0 and i != 0):
                    
                    if self.opts.save.save_pti:
                        with torch.no_grad():
                            y_hat_novel_pti, _ = self.forward(w_pivot, c_novel, eval=True)
                        
                        save_novel_pti_dir = os.path.join(self.opts.exp_dir, "novel_pti")
                        os.makedirs(save_novel_pti_dir, exist_ok=True)
                        save_novel_pti_path = os.path.join(save_novel_pti_dir, frame[0]+'.png')
                        cv2.imwrite(save_novel_pti_path, cv2.cvtColor(tensor2im(y_hat_novel_pti[0]), cv2.COLOR_RGB2BGR))

                    else:
                        with torch.no_grad():
                            generated_images, generated_depths = self.forward(w_pivot, pose, eval=True)
                            novel_images, novel_depths = self.forward(w_pivot, c_novel, eval=True)

                        dir_path = os.path.join(self.opts.exp_dir, frame[0])
                        os.makedirs(dir_path, exist_ok=True)

                        save_img_path = os.path.join(dir_path, 'ori.png')
                        cv2.imwrite(save_img_path, cv2.cvtColor(tensor2im(image[0]), cv2.COLOR_RGB2BGR))

                        save_pose_path = os.path.join(dir_path, 'c.pt')
                        torch.save(pose, save_pose_path)

                        save_rec_path = os.path.join(dir_path, f'{i:04d}_rec.png')
                        cv2.imwrite(save_rec_path, cv2.cvtColor(tensor2im(generated_images[0]), cv2.COLOR_RGB2BGR))

                        save_novel_path = os.path.join(dir_path, f'{i:04d}_novel.png')
                        cv2.imwrite(save_novel_path, cv2.cvtColor(tensor2im(novel_images[0]), cv2.COLOR_RGB2BGR))

                        save_depth = tensor2im(generated_depths[0].repeat(3, 1, 1), norm=True)
                        save_depth_path = os.path.join(dir_path, f'{i:04d}_depth.png')
                        cv2.imwrite(save_depth_path, cv2.cvtColor(save_depth, cv2.COLOR_RGB2BGR))
                        
                        if self.opts.save.save_video:
                            dir_path_out = os.path.join(dir_path, f'{i:04d}_frame')
                            os.makedirs(dir_path_out, exist_ok=True)
                            save_video_path = os.path.join(dir_path, f'{i:04d}.mp4')
                            video_out = imageio.get_writer(save_video_path, mode='I', fps=10, codec='libx264')
                            # save_video_path = os.path.join(dir_path, f'{i:04d}.gif')
                            # video_out = imageio.get_writer(save_video_path, format='GIF', mode='I', fps=10)
                            
                            for frame_idx, novel_pose in enumerate(self.render_poses):
                                with torch.no_grad():
                                    img, _ = self.forward(w_pivot, novel_pose, eval=True)
                                
                                if self.multi_views:
                                    # res1 = torch.cat((warp_imgs[frame_idx][0], masks[frame_idx][0].repeat(3, 1, 1)), dim=2)
                                    # res2 = torch.cat((teacher_imgs[frame_idx][0], img[0]), dim=2)
                                    # res = torch.cat((res1, res2), dim=1)
                                    res = torch.cat((image[0], inv_imgs[frame_idx][0], warp_imgs[frame_idx][0], masks[frame_idx][0].repeat(3, 1, 1), teacher_imgs[frame_idx][0], img[0]), dim=2)
                                    res = common.add_grid_lines(res, 512)
                                else:
                                    res = img[0]
                                res = tensor2im(res)

                                save_frame_path = os.path.join(dir_path_out, f'{frame_idx}.png')
                                cv2.imwrite(save_frame_path, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
                                video_out.append_data(res)

                            video_out.close()
                    
                    break

                
                use_ball_holder = self.training_step % self.opts.hyperparameters.locality_regularization_interval == 0

                # if self.use_wandb and log_images_counter % self.opts.log.image_rec_result_log_snapshot == 0:
                #     log_images_from_w([w_pivot], self.G, [image_name])

                self.training_step += 1
                log_images_counter += 1

            self.image_counter += 1


            if self.opts.save_finetune_model or self.opts.save_w_pivot:
                save_dir = os.path.join(self.opts.exp_dir, frame[0])
                os.makedirs(save_dir, exist_ok=True)
            if self.opts.save_finetune_model:
                torch.save(self.G, os.path.join(save_dir, f'model_{image_name}.pt'))
            if self.opts.save_w_pivot:
                torch.save(w_pivot, os.path.join(save_dir, f'w_pivot_{image_name}.pt'))

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