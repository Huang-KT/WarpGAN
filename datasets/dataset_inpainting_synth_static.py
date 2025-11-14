# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os
import numpy as np
import zipfile
import PIL.Image
from PIL import Image, ImageOps
import json
import torch
import dnnlib
import torchvision.transforms.functional as F
import random

from utils.depth_utils import read_pfm

try:
    import pyspng
except ImportError:
    pyspng = None

def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_data = self._load_raw_image(self._raw_idx[idx])
        src_img, src_c, codes_synth, src_hat, src_depth, target_img, target_c, target_hat, target_depth = raw_data[:9]
        fname = raw_data[-1]

        assert isinstance(src_img, np.ndarray)
        assert list(src_img.shape) == self.image_shape
        assert src_img.dtype == np.uint8
        
        def pre_img(image):
            image = torch.from_numpy(image / 255.0)
            # image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            image_resized = F.resize(image, (256, 256))
            return image, image_resized

        src_img, src_img_resized = pre_img(src_img)
        target_img, target_img_resized = pre_img(target_img)
        src_hat, _ = pre_img(src_hat)
        target_hat, _ = pre_img(target_hat)

        synth_src = src_img_resized, src_img, src_c, codes_synth, src_hat, src_depth
        synth_target = target_img_resized, target_img, target_c, target_hat, target_depth
        return_data = synth_src + synth_target + (fname,)

        return return_data

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

#----------------------------------------------------------------------------

class SynthImageFolderDataset(Dataset):
    def __init__(self,
        path_synth,              # Path to directory
        total_imgs_num = 200000,
        resolution     = None,     # Ensure specific resolution, None = highest available.
        **super_kwargs,          # Additional arguments for the Dataset base class.
    ):
        self._path_synth = path_synth

        if os.path.isdir(self._path_synth):
            self._type = 'dir'
            # self._all_fnames = {fname for fname in os.listdir(self._path) if os.path.isfile(os.path.join(self._path, fname))}
        elif self._file_ext(self._path_synth) == '.zip':
            self._type = 'zip'
            # self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        
        ###########################################################################
        self._image_dirs = [f'{img_idx:06d}' for img_idx in range(total_imgs_num)]
        ###########################################################################

        assert len(self._image_dirs) == total_imgs_num, 'Number of images does not match the specified total_imgs_num'

        if len(self._image_dirs) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path_synth))[0]
        raw_shape = [len(self._image_dirs)] + list(self._load_raw_image(0)[0].shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path_synth, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def read_img(self, fname):
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_image(self, raw_idx):
        img_dir = self._image_dirs[raw_idx]
        fname = img_dir

        src_img = self.read_img(os.path.join(img_dir, 'src.png'))
        src_c = torch.load(os.path.join(self._path_synth, img_dir, 'src_c.pt'), map_location='cpu')
        codes_synth = torch.load(os.path.join(self._path_synth, img_dir, 'codes.pt'), map_location='cpu')
        src_hat = self.read_img(os.path.join(img_dir, 'src_hat.png'))
        src_depth = torch.load(os.path.join(self._path_synth, img_dir, 'src_depth.pt'), map_location='cpu')

        target_img = self.read_img(os.path.join(img_dir, 'target.png'))
        target_c = torch.load(os.path.join(self._path_synth, img_dir, 'target_c.pt'), map_location='cpu')
        target_hat = self.read_img(os.path.join(img_dir, 'target_hat.png'))
        target_depth = torch.load(os.path.join(self._path_synth, img_dir, 'target_depth.pt'), map_location='cpu')

        return src_img, src_c, codes_synth, src_hat, src_depth, target_img, target_c, target_hat, target_depth, fname

#----------------------------------------------------------------------------
