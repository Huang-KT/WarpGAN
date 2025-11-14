import abc
import os
import pickle
from argparse import Namespace
import wandb
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from lpips import LPIPS

from training.projectors import w_plus_projector
from criteria import l2_loss
from criteria.localitly_regulizer import Space_Regulizer
from utils.models_utils import toogle_grad, load_old_G
from datasets.dataset_pti import ImageFolderDataset
from configs.paths_config import dataset_paths


class BaseCoach:
    def __init__(self, opts, run_name, multi_views, use_wandb):

        self.use_wandb = use_wandb
        self.w_pivots = {}
        self.image_counter = 0

        self.run_name = run_name
        self.opts = opts
        self.device = self.opts.device

        self.multi_views = multi_views
        if self.multi_views:
            print("Use Multi views for PTI!")
        else:
            print("Use Single views for PTI!")

        # if self.opts.hyperparameters.first_inv_type == 'w+':
        #     self.initilize_e4e()

        # self.e4e_image_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=self.opts.hyperparameters.lpips_type).to(self.device).eval()

        # self.restart_training()

        self.training_step = 1

        self.dataset = ImageFolderDataset(path=self.opts.data.path, resolution=None, use_labels=True, max_size=self.opts.data.max_size)
        print(f"Number of dataset: {len(self.dataset)}")
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G(self.device)
        toogle_grad(self.G, True)

        self.original_G = load_old_G(self.device)

        self.space_regulizer = Space_Regulizer(self.opts, self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()


    # def get_inversion(self, w_path_dir, image_name, image):
    #     embedding_dir = f'{w_path_dir}/{self.paths_config.pti_results_keyword}/{image_name}'
    #     os.makedirs(embedding_dir, exist_ok=True)

    #     w_pivot = None

    #     if self.opts.hyperparameters.use_last_w_pivots:
    #         w_pivot = self.load_inversions(w_path_dir, image_name)

    #     if not self.opts.hyperparameters.use_last_w_pivots or w_pivot is None:
    #         w_pivot = self.calc_inversions(image, image_name)
    #         torch.save(w_pivot, f'{embedding_dir}/0.pt')

    #     w_pivot = w_pivot.to(self.device)
    #     return w_pivot

    # def load_inversions(self, w_path_dir, image_name):
    #     if image_name in self.w_pivots:
    #         return self.w_pivots[image_name]

    #     if self.opts.hyperparameters.first_inv_type == 'w+':
    #         w_potential_path = f'{w_path_dir}/{self.paths_config.e4e_results_keyword}/{image_name}/0.pt'
    #     else:
    #         w_potential_path = f'{w_path_dir}/{self.paths_config.pti_results_keyword}/{image_name}/0.pt'
    #     if not os.path.isfile(w_potential_path):
    #         return None
    #     #breakpoint()
    #     w = torch.load(w_potential_path).to(self.device)
    #     self.w_pivots[image_name] = w
    #     return w

    def calc_inversions(self, image, pose, image_name, initial_w=None):
        
        if self.opts.hyperparameters.first_inv_type == 'w+':
            w = self.get_e4e_inversion(image)

        else:
            id_image = torch.squeeze((image.to(self.device) + 1) / 2) * 255
            pose = pose.to(self.device)
            w = w_plus_projector.project(self.G, id_image,pose, device=torch.device(self.device), w_avg_samples=600,
                                         num_steps=self.opts.hyperparameters.first_inv_steps, w_name=image_name,
                                         use_wandb=self.use_wandb,
                                         initial_w=initial_w, first_inv_lr=self.opts.hyperparameters.first_inv_lr,)

        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=self.opts.hyperparameters.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if self.opts.hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=self.training_step)
            loss += l2_loss_val * self.opts.hyperparameters.pt_l2_lambda
        if self.opts.hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=self.training_step)
            loss += loss_lpips * self.opts.hyperparameters.pt_lpips_lambda

        if use_ball_holder and self.opts.hyperparameters.use_locality_regularization:
            # ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, pose, use_wandb=self.use_wandb)
            # loss += ball_holder_loss_val
            assert False, "Space regularization is not implemented yet"

        return loss, l2_loss_val,loss_lpips

    def forward(self, w, pose, eval):
        if eval==True:
            self.G.eval()
        else:
            self.G.train()
        generated= self.G.synthesis(w, pose, noise_mode='const', force_fp32=True)
        generated_images = generated['image']
        generated_depths = generated['image_depth']
        return generated_images, generated_depths

    # def initilize_e4e(self):
    #     ckpt = torch.load(self.paths_config.e4e, map_location='cpu')
    #     opts = ckpt['opts']
    #     opts['batch_size'] = hyperparameters.train_batch_size
    #     opts['checkpoint_path'] = self.paths_config.e4e
    #     opts = Namespace(**opts)
    #     self.e4e_inversion_net = pSp(opts)
    #     self.e4e_inversion_net.eval()
    #     self.e4e_inversion_net = self.e4e_inversion_net.to(global_config.device)
    #     toogle_grad(self.e4e_inversion_net, False)

    # def get_e4e_inversion(self, image):
    #     image = (image + 1) / 2
    #     new_image = self.e4e_image_transform(image[0]).to(global_config.device)
    #     _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
    #                                   input_code=False)
    #     if self.use_wandb:
    #         log_image_from_w(w, self.G, 'First e4e inversion')
    #     return w
