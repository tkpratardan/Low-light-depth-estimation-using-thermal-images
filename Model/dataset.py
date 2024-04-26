# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 0.0 # 1 / max_depth
    max_disp = 256.0 # 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        # self.to_tensor = lambda x: x
        # self.to_tensor = transforms.Normalize(0.5, 0.5)

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        self.thermal_resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.thermal_resize[i] = (1, self.height // s, self.width // s)
            self.resize[i] = transforms.Resize((self.height // s, self.width // s))
                                               #interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

            if "thermal" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = F.interpolate(inputs[(n, im, i - 1)][None, None, ...], self.thermal_resize[i])[0, 0]

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("thermal", <frame_id>, <scale>)        for raw thermal images (if any),
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 1.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        frame_index = index # int(line[1])
        side = None

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index, side, do_flip, i)
            exists, thermal = self.get_thermal(folder, frame_index, side, do_flip, i)
            if exists:
                inputs[("thermal", i, -1)] = thermal

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            del inputs[("thermal", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_thermal(self, folder, frame_index, side, do_flip):
        return False, None

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

class ThermalDataset(MonoDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    THERMAL_MEAN = 0.8364310589802526
    THERMAL_STD = 0.18558015690667787

    def __init__(self, *args, **kwargs):
        super(ThermalDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.image_width = 4096
        self.image_height = 3000
        self.K = np.array([[4643.03812, 0, 2071.019842, 0],
                           [0, 4643.03812, 1497.908013, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.K[0] /= self.image_width
        self.K[1] /= self.image_height
        self.color_shape = None
        self.min_depth = 0.1
        self.max_depth = 100.0


        mask = []
        for i, f in enumerate(self.filenames):
            if i == 0 or i == len(self.filenames) - 1:
                continue
            frame_num = int(f.split(" ")[1])
            prev_frame = int(self.filenames[i - 1].split(" ")[1])
            next_frame = int(self.filenames[i + 1].split(" ")[1])

            if (frame_num - prev_frame < 10) and (next_frame - frame_num < 10):
                mask.append(i)

        self.all_files = [f for f in self.filenames]
        self.filenames = np.array(self.filenames)[mask].tolist()
        self.filename_index_table = {
            i : mask[i] for i in range(len(self.filenames))
        }

        print(f"Total files with nearby frames: {len(self.filenames)}/{len(self.all_files)}")


    def get_color(self, folder, frame_index, side, do_flip, offset):
        all_file_index = self.filename_index_table[frame_index]
        fname = self.all_files[all_file_index + offset].split(" ")[0]
        rgb = np.load(f'{self.data_path}/{fname}_rgb.npy')
        self.color_shape = rgb.shape

        rgb = pil.fromarray(rgb, 'RGB')
        if do_flip:
            rgb = rgb.transpose(pil.FLIP_LEFT_RIGHT)

        return rgb


    def get_thermal(self, folder, frame_index, side, do_flip, offset):
        all_file_index = self.filename_index_table[frame_index]
        fname = self.all_files[all_file_index + offset].split(" ")[0]
        thermal_norm = self.load_thermal_data(f'{self.data_path}/{fname}_thermal.npy')

        if do_flip:
            thermal_norm = torch.flip(thermal_norm, dims=(-1,))

        return True, thermal_norm

    @staticmethod
    def load_thermal_data(fname):
        thermal = np.load(fname).astype(np.uint16)
        thermal_values = thermal[thermal > 0]
        min_value = np.percentile(thermal_values, 5) * 1.0
        max_value = np.percentile(thermal_values, 90) * 1.0

        thermal_norm = (np.clip(thermal, min_value, max_value) - min_value) / (max_value - min_value)
        thermal_norm = torch.from_numpy(thermal_norm.astype(np.float32))[None, ...]

        thermal_norm_unit = (thermal_norm - thermal_norm.mean()) / thermal_norm.std()
        thermal_norm = (thermal_norm_unit + 0.5)

        return thermal_norm


    def check_depth(self):
        return True


    def get_depth(self, folder, frame_index, side, do_flip):
        all_file_index = self.filename_index_table[frame_index]
        fname = self.all_files[all_file_index].split(" ")[0]
        depth_gt = np.load(f'{self.data_path}/{fname}_gt.npy')#[0, 0]
        depth_gt = skimage.transform.resize(depth_gt, self.color_shape[:-1], mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt # disp_to_depth(depth_gt, self.min_depth, self.max_depth)[1]
