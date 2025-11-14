import os
import hydra
from omegaconf import DictConfig, OmegaConf
import random
from string import ascii_uppercase
import numpy as np
import torch
import wandb

import sys
sys.path.append(".")
sys.path.append("..")

from training.pti.multi_id_coach import MultiIDCoach
from training.pti.single_id_coach import SingleIDCoach
from utils.common import addtime2path


SEED = 2107
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


def run_PTI(opts, run_name='', multi_views=False, use_wandb=False, use_multi_id_training=False):

    if run_name == '':
        run_name = ''.join(random.choice(ascii_uppercase) for _ in range(12))

    if use_wandb:
        wandb.init(project="PTI", reinit=True, name=run_name)

    if use_multi_id_training:
        coach = MultiIDCoach(opts, run_name, multi_views, use_wandb)
    else:
        coach = SingleIDCoach(opts, run_name, multi_views, use_wandb)

    coach.train()

    return run_name


@hydra.main(config_path="../configs", config_name="pti")
def main(opts: DictConfig):

    # opts.exp_dir = addtime2path(opts.exp_dir)

    if os.path.exists(opts.exp_dir):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    os.makedirs(opts.exp_dir, exist_ok=False)

    print(OmegaConf.to_yaml(opts))
    OmegaConf.save(opts, os.path.join(opts.exp_dir, 'config.yaml'))

    run_PTI(opts, run_name=opts.exp_dir, multi_views=opts.multi_views, use_wandb=opts.log.use_wandb)


if __name__ == '__main__':
    main()