dataset_paths = {
	'test': './data/celeba-hq_1000_align',
	'train': './data/FFHQ-LPFF-EG3D_all',
}

dataset_static_paths = {
	'test': './data/celeba-hq_1000_static_rebalanced',
	'train': './data/FFHQ-EG3D_all_static_rebalanced',
	'synth': './data/SynthData100000_rebalanced',
}

model_paths = {
	'eg3d_ffhq_pth': './pretrained_models/ffhq/ffhq512-128.pth',
    'discriminator': './pretrained_models/ffhq/discriminator.pth',
    
	'eg3d_ffhq': './pretrained_models/eg3d/ffhq/ffhq512-128.pkl',
	'latent_avg': './pretrained_models/eg3d/ffhq/latent_avg.pt',
	'eg3d_ffhq_rebalanced': './pretrained_models/eg3d/ffhq/ffhqrebalanced512-128.pkl',
	'latent_avg_rebalanced': './pretrained_models/eg3d/ffhq/latent_avg_rebalanced.pt',
    'eg3d_ffhq_lpff': './pretrained_models/eg3d/ffhq_lpff/var1-128.pkl',
	'latent_avg_plus': './pretrained_models/eg3d/ffhq_lpff/latent_avg_plus.pt',
    'eg3d_ffhq_lpff_rebalanced': './pretrained_models/eg3d/ffhq_lpff/var2-128.pkl',
	'latent_avg_plus_rebalanced': './pretrained_models/eg3d/ffhq_lpff/latent_avg_plus_rebalanced.pt',
    
	'ir_se50': './pretrained_models/model_ir_se50.pth',
	'goae': './pretrained_models/inversion/goae_FFHQ_ori.pt',
	'moco': './pretrained_models/moco_v2_800ep_pretrain.pt',
}
