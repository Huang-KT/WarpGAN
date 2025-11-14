import os
from argparse import Namespace
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import glob
from PIL import Image
import imageio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import transforms

import sys
sys.path.append(".")
sys.path.append("..")

from utils.models_utils import load_tuned_G, load_old_G
from editings.latent_editor import LatentEditor
from models.wplusnet import WplusNet
from models.eg3d.camera_utils import LookAtPoseSampler
from models.saicinpainting.utils import set_requires_grad
from editings.poi_ganspace.run_ganspace import edit_w_ganspace
from configs.paths_config import model_paths
from editings.CLIPStyle.mapper.styleclip_mapper import StyleCLIPMapper


SEED = 2107
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


class Editor:
    def __init__(self, opts):
        self.opts = opts
        self.device = self.opts.device

        if self.opts.editing.type == 'interfacegan':
            self.editor = LatentEditor()
            direction_path = getattr(self.opts.editing.interfacegan, self.opts.editing.interfacegan.att)
            self.direction = torch.from_numpy(np.load(direction_path)).to(self.device).float()
            # self.direction = torch.load(direction_path).to(self.device).float()  # cat
            print("Load Direction from", direction_path)

        # elif self.opts.editing.type == 'ganspace':
        #     self.pca_comp = np.load(self.opts.editing.ganspace.pca_comp)
        #     print("Load PCA Component from", self.opts.editing.ganspace.pca_comp)
        #     # idx_comp, start_layer, layer_num, edit_power 
        #     self.ganspace_directions = {
        #         'bright hair': (2, 7, 7, 4), #positive (direction)
        #         'smile': (12, 0, 5, 2), #positive 
        #         'age' : (5, 0, 5, 3.5), #negative: young
        #         'short hair': (2, 0, 5, 4), #negative
        #         'glass': (4, 0, 5, 4), #negative
        #         'gender': (0, 0, 5, 4) #negative(female -> male)
        #     }

        elif self.opts.editing.type == 'styleclip':
            self.editor = LatentEditor()
            ckpt_styleclip_path = getattr(self.opts.editing.styleclip, self.opts.editing.styleclip.att)
            ckpt_styleclip = torch.load(ckpt_styleclip_path, map_location='cpu')
            opts_styleclip = ckpt_styleclip['opts']
            opts_styleclip['checkpoint_path'] = ckpt_styleclip_path
            opts_styleclip.update(vars(self.opts.editing.styleclip))
            opts_styleclip = Namespace(**opts_styleclip)
            self.styleclip_mapper = StyleCLIPMapper(opts_styleclip, self.G, load_G=False).eval().to(self.device)
        else:
            raise ValueError("Unknown editing type")

    def edit_w(self, w):
        if self.opts.editing.type == 'interfacegan':
            w_edit = self.editor.apply_interfacegan(w, self.direction, factor=self.opts.editing.interfacegan.factor)
        # elif self.opts.editing.type == 'ganspace':
        #     idx_comp, start_layer, layer_num, edit_power = self.ganspace_directions[self.opts.editing.ganspace.att]
        #     w_edit = edit_w_ganspace(self.pca_comp, w, idx_comp, start_layer, layer_num, edit_power)
        elif self.opts.editing.type == 'styleclip':
            w_edit = self.editor.apply_styleclip(w, self.styleclip_mapper, factor_step=self.opts.editing.styleclip.factor)
        else:
            raise ValueError("Unknown editing type")
        return w_edit

    def forward_G(self, G, w, pose, eval):
        if eval == True:
            G.eval()
        else:
            G.train()
        generated = G.synthesis(w, pose, noise_mode='const', force_fp32=True)
        generated_images = generated['image']
        generated_depths = generated['image_depth']
        return generated_images, generated_depths

    def forward(self, name):

        data_path = os.path.join(self.opts.pti_path, name)
        # Load G
        G_path = os.path.join(data_path, f'model_{name}.pt')

        self.G = load_tuned_G(G_path, self.device)
        # self.G = load_old_G(self.device, model_paths['eg3d_ffhq'])
        # print("Load G from", G_path)

        w_path = os.path.join(data_path, f'w_pivot_{name}.pt')
        
        img_256, img, c = self.read_data(data_path, self.opts.device)
        
        # if self.opts.get_w_pivot == 'encoder':
        #     with torch.no_grad():
        #         w_pivot = self.gan.encoder_forward(img_256)
        # elif self.opts.get_w_pivot == 'opt':
        #     w_pivot = torch.load(self.w_path).to(self.device)
        # else:
        #     raise ValueError("Unknown w_pivot type")
        w_pivot = torch.load(w_path).to(self.device)

        w_edit = self.edit_w(w_pivot)
        # print("Edit Done")

        synth_type = 'video'  # 'video' or 'image'

        if synth_type == 'image':
            yaw = -np.pi/6
            c_novel = self.get_pose(cam_pivot=torch.tensor([0, 0, 0.2]).to(self.device),
                                    intrinsics=torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).to(self.device),
                                    yaw=yaw, pitch=0).repeat(w_edit.shape[0], 1)  # b, 25

            with torch.no_grad():
                res_ori, _ = self.forward_G(self.G, w_pivot, c, eval=True)
                res_novel, _ = self.forward_G(self.G, w_pivot, c_novel, eval=True)
                res_edit, _ = self.forward_G(self.G, w_edit, c, eval=True)
                res_edit_novel, _ = self.forward_G(self.G, w_edit, c_novel, eval=True)

            save_res_dir = os.path.join(self.opts.exp_dir, self.opts.editing.interfacegan.att)
            os.makedirs(save_res_dir, exist_ok=True)

            save_res_single_dir = os.path.join(save_res_dir, "single")
            os.makedirs(save_res_single_dir, exist_ok=True)
            save_res_cat_dir = os.path.join(save_res_dir, "cat")
            os.makedirs(save_res_cat_dir, exist_ok=True)
            
            save_path = os.path.join(save_res_single_dir, name+".png")
            cv2.imwrite(save_path, cv2.cvtColor(self.tensor2im(res_ori[0]), cv2.COLOR_RGB2BGR))

            save_novel_path = os.path.join(save_res_single_dir, name+"_novel.png")
            cv2.imwrite(save_novel_path, cv2.cvtColor(self.tensor2im(res_novel[0]), cv2.COLOR_RGB2BGR))

            save_edit_path = os.path.join(save_res_single_dir, name+"_edit.png")
            cv2.imwrite(save_edit_path, cv2.cvtColor(self.tensor2im(res_edit[0]), cv2.COLOR_RGB2BGR))

            save_edit_novel_path = os.path.join(save_res_single_dir, name+"_edit_novel.png")
            cv2.imwrite(save_edit_novel_path, cv2.cvtColor(self.tensor2im(res_edit_novel[0]), cv2.COLOR_RGB2BGR))

            cat_res = torch.cat([img[0], res_ori[0], res_novel[0], res_edit[0], res_edit_novel[0]], dim=2)
            save_cat_path = os.path.join(save_res_cat_dir, name+".png")
            cv2.imwrite(save_cat_path, cv2.cvtColor(self.tensor2im(cat_res), cv2.COLOR_RGB2BGR))


        elif synth_type == 'video':
            self.multi_view(w_pivot, w_edit, self.opts.exp_dir, name)
        else:
            raise ValueError("Unknown synthesis type")
        # print("Synthesis Done")

    def synth(self, w_edit):
        yaw_list = [-np.pi/6, -np.pi/8, 0, np.pi/8, np.pi/6]
        res_edit = []
        for yaw in yaw_list:
            novel_pose = self.get_pose(cam_pivot=torch.tensor([0, 0, 0.2]).to(self.device),
                                       intrinsics=torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).to(self.device),
                                       yaw=yaw, pitch=0).repeat(w_edit.shape[0], 1)  # b, 25

            with torch.no_grad():
                img_edit, _ = self.forward_G(self.G, w_edit, novel_pose, eval=True)
            
            res_edit.append(img_edit)
        
        return res_edit

    def multi_view(self, w, w_edit, save_path, name):
        if self.opts.editing.type == 'interfacegan':
            video_name = name + '_' + self.opts.editing.interfacegan.att + '_' + str(self.opts.editing.interfacegan.factor) + '.mp4'
        elif self.opts.editing.type == 'ganspace':
            video_name = name + '_ganspace_' + self.opts.editing.ganspace.att + '.mp4'
        elif self.opts.editing.type == 'styleclip':
            video_name = name + '_styleclip_' + self.opts.editing.styleclip.att + '_' + str(self.opts.editing.styleclip.factor) + '.mp4'
        else:
            raise ValueError("Unknown editing type")

        save_video_path = os.path.join(save_path, video_name)
        video_out = imageio.get_writer(save_video_path, mode='I', fps=10, codec='libx264')

        cam_pivot = torch.Tensor([0., 0., 0.2]).to(self.device)
        pitch_range, yaw_range = 0.25, 0.35
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=self.device)
        num_keyframes = 30
        # render_poses = []
        for frame_idx in tqdm(range(num_keyframes)):
            cam2world_pose = LookAtPoseSampler.sample(3.14/2        + yaw_range   * np.sin(2 * 3.14 * frame_idx / (num_keyframes)),
                                                      3.14/2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes)),
                                                      cam_pivot, radius=2.7, device=self.device)
            novel_pose = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            with torch.no_grad():
                img_ori, _ = self.forward_G(self.G, w, novel_pose, eval=True)
                img_edit, _ = self.forward_G(self.G, w_edit, novel_pose, eval=True)
            img = self.tensor2im(torch.cat([img_ori[0], img_edit[0]], dim=2))
            video_out.append_data(img)
        video_out.close()

        print(f"Save video to {save_video_path}")

    def read_data(self, path, device):
        img_path = os.path.join(path, 'ori.png')
        image = Image.open(img_path).convert('RGB')
        # image = np.array(image)
        # image = torch.from_numpy(image / 255.0)
        # image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        source_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = source_transform(image)
        image = image.unsqueeze(0).to(device).float()
        image_resized = F.resize(image, (256, 256))

        c_path = os.path.join(path, 'c.pt')
        c = torch.load(c_path).to(device).float()

        return image_resized, image, c

    def get_pose(self, cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7):
        
        if yaw is None:
            yaw = np.random.uniform(-yaw_range, yaw_range)
        if pitch is None:
                pitch = np.random.uniform(-pitch_range, pitch_range)

        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=cam_pivot.device)
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


@hydra.main(config_path="../configs", config_name="editing")
def main(opts: DictConfig):

    opts.exp_dir = os.path.join(opts.exp_dir)
    os.makedirs(opts.exp_dir, exist_ok=True)

    editor = Editor(opts)

    name_list = [name for name in os.listdir(opts.pti_path) if os.path.isdir(os.path.join(opts.pti_path, name))]
    name_list = sorted(name_list)
    for name in tqdm(name_list):
        # print(f"Processing {name}...")
        editor.forward(name)


if __name__ == '__main__':
    main()