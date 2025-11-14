#!/bin/bash
conda create -n warpgan python=3.9
conda activate warpgan

# gan inversion
pip install \
    numpy==1.22.4 \
    click==8.0.4 \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    Pillow==9.3.0 \
    scipy==1.7.1 \
    requests==2.26.0 \
    tqdm==4.62.2 \
    matplotlib==3.4.2 \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.3 \
    imgui==1.3.0 \
    glfw==2.2.0 \
    PyOpenGL==3.1.5 \
    pyspng==0.1.1 \
    psutil==5.9.4 \
    mrcfile==1.4.3 \
    tensorboard==2.9.1 \
    gdown==4.7.1 \
    opencv-python==4.6.0.66 \
    kornia==0.6.8 \
    mtcnn==0.1.1 \
    dominate==2.7.0 \
    scikit-image==0.19.3 \
    tensorflow==2.9.2 \
    trimesh==3.16.2 \
    ninja==1.11.1 \
    wandb==0.13.5 \
    pytorch-msssim==0.2.1 \
    plyfile==0.7.4 \
    --extra-index-url https://download.pytorch.org/whl/cu113

## lama
pip install \
    albumentations==0.5.2 \
    hydra-core==1.1.0 \
    webdataset \
    easydict==1.9.0 \
    scikit-learn==0.24.2 \
    pandas \
    torchmetrics==0.6.0 \
    pytorch-lightning==1.4.4 \

## warp
pip install git+https://github.com/pesser/splatting@1427d7c4204282d117403b35698d489e0324287f#egg=splatting

## goae
pip install \
    timm==0.6.12 \
    yacs