import pickle
import functools
import torch
# from configs import global_config
from configs.paths_config import model_paths


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# def load_tuned_G(run_id, type, device):
#     new_G_path = f'{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt'
#     with open(new_G_path, 'rb') as f:
#         new_G = torch.load(f).to(device).eval()
#     new_G = new_G.float()
#     toogle_grad(new_G, False)
#     return new_G

def load_tuned_G(new_G_path, device):
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G(device, path=None):
    if path is None:
        path = model_paths["eg3d_ffhq_rebalanced"]
    print("Loading old G from", path)
    with open(path, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(device).eval()
        old_G = old_G.float()
    return old_G
