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
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        load_conf_map = False,
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self._use_conf_map = load_conf_map

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
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
        # image, fname = self._load_raw_image(self._raw_idx[idx])
        # fname = fname[:-4]
        raw_data = self._load_raw_image(self._raw_idx[idx])
        image, code, y_hat, depth, y_hat_mirror, depth_mirror, c_novel, y_hat_novel, depth_novel = raw_data[:9]
        fname = raw_data[-1][:-4]
        
        if self._use_conf_map:
            mirror_conf_map = torch.from_numpy(self._load_conf_map(self._raw_idx[idx]))
        else:
            mirror_conf_map = ['not loaded']

        mirror_image = np.array(ImageOps.mirror(Image.fromarray(image.transpose(1, 2, 0)))).transpose(2, 0, 1)

        label = self.get_label(idx)

        pose, intrinsics = np.array(label[:16]).reshape(4,4), np.array(label[16:]).reshape(3, 3)
        flipped_pose = flip_yaw(pose)
        mirror_label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)])

        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        
        def pre_img(image):
            image = torch.from_numpy(image / 255.0)
            # image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            image_resized = F.resize(image, (256, 256))
            return image, image_resized
        
        image, image_resized = pre_img(image)
        mirror_image, mirror_image_resized = pre_img(mirror_image)
        y_hat, _ = pre_img(y_hat)
        y_hat_mirror, _ = pre_img(y_hat_mirror)
        y_hat_novel, _ = pre_img(y_hat_novel)

        real_data = image_resized, image, torch.from_numpy(label), code, y_hat, depth
        mirror_data = mirror_image_resized, mirror_image, torch.from_numpy(mirror_label), y_hat_mirror, depth_mirror, mirror_conf_map
        novel_data = c_novel, y_hat_novel, depth_novel
        return_data = real_data + mirror_data + novel_data + (fname,)
        return return_data

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

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

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution   = None, # Ensure specific resolution, None = highest available.
        datast_json  = None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            # self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
            self._all_fnames = {fname for fname in os.listdir(self._path) if os.path.isfile(os.path.join(self._path, fname))}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()

        self.dataset_json = 'dataset.json' if datast_json is None else datast_json
        
        ###############################################################################
        # with open(os.path.join(path, self.dataset_json), 'r') as f:
        #     self._image_fnames = json.load(f)
        # self._image_fnames = [x[0] for x in self._image_fnames['labels']]
        # self._image_fnames = sorted(self._image_fnames)
        with open(os.path.join(path, self.dataset_json), 'r') as f:
            self._image_fnames = json.load(f)
        
        self._image_dirs = [x[0][:-4] for x in self._image_fnames['labels']]
        self._image_dirs = sorted(self._image_dirs)

        self._image_fnames = [x[0] for x in self._image_fnames['labels']]
        self._image_fnames = sorted(self._image_fnames)
        ###############################################################################

        self.confmap_fnames = [f'{p.split("/")[-1][:-4]}.npy' for p in self._image_fnames]
        self.confmap_root = os.path.join(self._path, 'conf_map')
            

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0)[0].shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        if self._use_conf_map:
            assert len(self.confmap_fnames) == len(self._image_fnames)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname, base_path=None):
        base_path = self._path if base_path is None else base_path
        if self._type == 'dir':
            return open(os.path.join(base_path, fname), 'rb')
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

    def read_img(self, fname, base_path=None):
        base_path = self._path if base_path is None else base_path
        with self._open_file(fname, base_path) as f:
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
        fname = img_dir + '.png'

        x = self.read_img(os.path.join(img_dir, 'x.png'))
        code = torch.load(os.path.join(self._path, img_dir, 'codes.pt'), map_location='cpu')
        y_hat = self.read_img(os.path.join(img_dir, 'y_hat.png'))
        depth = torch.load(os.path.join(self._path, img_dir, 'depth.pt'), map_location='cpu')

        y_hat_mirror = self.read_img(os.path.join(img_dir, 'y_hat_mirror.png'))
        # depth_mirror = torch.load(os.path.join(self._path, img_dir, 'depth_mirror.pt'), map_location='cpu')
        depth_mirror = torch.flip(depth, dims=[2])

        novel_view = random.randint(1, 3)
        c_novel = torch.load(os.path.join(self._path, img_dir, f'c_novel_{novel_view}.pt'), map_location='cpu')
        y_hat_novel = self.read_img(os.path.join(img_dir, f'y_hat_novel_{novel_view}.png'))
        depth_novel = torch.load(os.path.join(self._path, img_dir, f'depth_novel_{novel_view}.pt'), map_location='cpu')

        return_data = x, code, y_hat, depth, y_hat_mirror, depth_mirror, c_novel, y_hat_novel, depth_novel, fname

        return return_data

    def _load_conf_map(self, raw_idx):
        fname = self.confmap_fnames[raw_idx]
        with open(os.path.join(self.confmap_root, fname), 'rb') as f:
            conf_map = np.load(f)
        conf_map = Image.fromarray(conf_map)
        conf_map = np.array(ImageOps.mirror(conf_map))
        return conf_map

    def _load_raw_labels(self):
        # fname = 'dataset.json'
        fname = self.dataset_json
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
