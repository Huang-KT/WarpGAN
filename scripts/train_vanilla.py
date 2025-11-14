import os
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import torch

import sys
sys.path.append(".")
sys.path.append("..")

from training.coach_vanilla import Coach
from utils.common import addtime2path


SEED = 2107
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


@hydra.main(config_path="../configs", config_name="train_vanilla")
def main(opts: DictConfig):

	opts.exp_dir = addtime2path(opts.exp_dir)

	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir, exist_ok=True)

	print(OmegaConf.to_yaml(opts))
	OmegaConf.save(opts, os.path.join(opts.exp_dir, 'config.yaml'))

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
