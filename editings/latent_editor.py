import torch

from configs import paths_config
from editings import ganspace
# from utils.data_utils import tensor2im


class LatentEditor(object):

    def apply_ganspace(self, latent, ganspace_pca, edit_directions):
        edit_latents = ganspace.edit(latent, ganspace_pca, edit_directions)
        return edit_latents

    def apply_interfacegan(self, latent, direction, factor=1, factor_range=None):
        edit_latents = []
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latent + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent + factor * direction
        return edit_latents

    # def apply_styleclip(self, inputs, net, couple_outputs=False, factor_step=0.1)
    #     w, c = inputs
    #     with torch.no_grad():
    #         w_hat = w + factor_step * net.mapper(w)
    #         x_hat = net.decoder.synthesis(w_hat, c, noise_mode='const')['image']
    #         result_batch = (x_hat, w_hat)
    #         if couple_outputs:
    #             x = net.decoder.synthesis(w, c, noise_mode='const')['image']
    #             result_batch = (x_hat, w_hat, x)
    #     return result_batch
    def apply_styleclip(self, w, net, factor_step=0.1):
        with torch.no_grad():
            w_edit = w + factor_step * net.mapper(w)
        return w_edit