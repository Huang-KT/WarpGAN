# WarpGAN: Warping-Guided 3D GAN Inversion with Style-Based Novel View Inpainting
### Accepted to NeurIPS 2025

[Paper](https://arxiv.org/abs/2511.08178) | [Project Website](https://huang-kt.github.io/warpgan-project/)


<img src="assets/teaser.gif"/>


## Requirements

```sh
source ./scripts/install_deps.sh
```


## Checkpoints
We have uploaded the pre-trained models required for image preprocessing, training, inference, and editing to [Google Drive](https://drive.google.com/drive/folders/1G9PeyrCS1gTyF3C957l6Key__t44aiIE?usp=sharing). After downloading, please place them in the corresponding directories (`./pose_estimation`, `./pretrained_models`, `./editings`).




## Dataset preparation

### Training Dataset
Download the [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset) as the training dataset; optionally extend it with the [LPFF Dataset](https://github.com/oneThousand1000/LPFF-dataset).

Follow [EG3D](https://github.com/NVlabs/ffhq-dataset) or [LPFF](https://github.com/oneThousand1000/LPFF-dataset/tree/master/data_processing) for pose extraction and face alignment,
or simply download the ready-to-use official releases of both datasets ([FFHQ Dataset](https://drive.google.com/file/d/1XP0od0OM_pdPE6Cnv3joMYw7pIzHDEqo/view?usp=sharing) and [LPFF Dataset](https://huggingface.co/datasets/onethousand/LPFF/tree/main/LPFF-dataset)).

### Testing Dataset
We follow [HFGI](https://github.com/jiaxinxie97/HFGI3D/) to preprocess customized images.
```sh
cd ./pose_estimation
python extract_pose.py 0 ori_data tmp_data align_data
```



## Training

### Training 3DGAN Inversion Encoder
Run the following command to train the encoder. Config file is `configs/train_vanilla.yaml`.
```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train_vanilla.py
```

### Generating Static Dataset
Leverage synthetic images sampled from the 3D GAN to assist in training SVINet. Besides, use the encoder trained above to reconstruct these images and generate novel views beforehand, accelerating SVINet training.
```sh
CUDA_VISIBLE_DEVICES=0 python scripts/gen_synthimg.py
```

Employ the encoder already trained to reconstruct real images and generate novel views beforehand, accelerating SVINet training.
```sh
CUDA_VISIBLE_DEVICES=0 python scripts/gen_novelview.py
```

### Training SVINet
Run the following command to train SVINet. Config file is `configs/train_inpainting.yaml`.
```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train_inpainting.py
```

## Inference
Run the following command to synthesize novel-view images from the input. Config file is `configs/infer.yaml`.
```sh
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py
```

## PTI
Use novel-view images synthesized by WarpGAN to assist [PTI](https://github.com/danielroich/PTI) training. Config file is `configs/pti.yaml`.
```sh
CUDA_VISIBLE_DEVICES=0 python ./scripts/run_pti.py
```

## Editing
Perform editing with the latent code and fine-tuned generator obtained from PTI. Config file is `configs/editing.yaml`.
```sh
CUDA_VISIBLE_DEVICES=0 python scripts/editing_ptiG.py
```


## Acknowlegement
We thank the authors of [EG3D](https://github.com/NVlabs/eg3d), [LPFF](https://github.com/oneThousand1000/LPFF-dataset), [Triplanenet](https://github.com/anantarb/triplanenet), [GOAE](https://github.com/jiangyzy/GOAE), [PTI](https://github.com/danielroich/PTI), [HFGI](https://github.com/jiaxinxie97/HFGI3D/), [LaMa](https://github.com/advimman/lama) and [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) for sharing their code.

## Citation
```
@misc{huang2025warpganwarpingguided3dgan,
      title={WarpGAN: Warping-Guided 3D GAN Inversion with Style-Based Novel View Inpainting}, 
      author={Kaitao Huang and Yan Yan and Jing-Hao Xue and Hanzi Wang},
      year={2025},
      eprint={2511.08178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.08178}, 
}
```
