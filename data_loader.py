import json
import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision import transforms as T

import config as cfg

IMAGE_TRAIN_TRANSFORMS = T.Compose([
    # image_rescale_zero_to_1_transform(),
    T.ToPILImage(),
    T.Resize(cfg.RESIZE),
    T.RandomCrop(cfg.CROP_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
IMAGE_TEST_TRANSFORMS = T.Compose([
    # image_rescale_zero_to_1_transform(),
    T.ToPILImage(),
    T.Resize(cfg.RESIZE),
    T.CenterCrop(cfg.CROP_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
Y_TRANSFORMS = torch.FloatTensor


def get_dataset(dset_name, *args, **kwargs):
    train_ds, val_ds, test_ds = None, None, None

    if dset_name == "lamem":
        train_ds = LaMemLoader(split="train",
                               transform=IMAGE_TRAIN_TRANSFORMS,
                               target_transform=Y_TRANSFORMS)
        val_ds = LaMemLoader(split="val",
                             transform=IMAGE_TEST_TRANSFORMS,
                             target_transform=Y_TRANSFORMS)
        test_ds = LaMemLoader(split="test",
                              transform=IMAGE_TEST_TRANSFORMS,
                              target_transform=Y_TRANSFORMS)
    elif dset_name == "memento_frames":
        train_ds = MementoFramesLoader(split='train',
                                       transform=IMAGE_TRAIN_TRANSFORMS,
                                       target_transform=Y_TRANSFORMS)
        val_ds = MementoFramesLoader(split='val',
                                     transform=IMAGE_TEST_TRANSFORMS,
                                     target_transform=Y_TRANSFORMS)
        test_ds = MementoFramesLoader(split='test',
                                      transform=IMAGE_TEST_TRANSFORMS,
                                      target_transform=Y_TRANSFORMS)
    else:
        raise RuntimeError("Unrecognized dset name: {}".format(dset_name))

    return train_ds, val_ds, test_ds


class ImageTxtInstance(data.Dataset):
    def __init__(self,
                 root,
                 txtFile,
                 loader=datasets.folder.default_loader,
                 transform=None,
                 target_transform=None):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        with open(os.path.join(txtFile)) as infile:
            self.samples = [line.strip().split()[:2] for line in infile]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fname, target = self.samples[index]
        path = os.path.join(self.root, fname)
        sample = np.array(self.loader(path))

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class LaMemLoader(ImageTxtInstance):
    def __init__(self,
                 split,
                 lamem_root=cfg.LAMEM_ROOT,
                 transform=None,
                 target_transform=None):
        images_root = os.path.join(cfg.LAMEM_ROOT, "images")
        txtFile = os.path.join(cfg.LAMEM_ROOT, "splits",
                               "{}_1.txt".format(split))
        super(LaMemLoader, self).__init__(root=images_root,
                                          txtFile=txtFile,
                                          transform=transform,
                                          target_transform=None)
        self.lamem_target_transform = target_transform

    def __getitem__(self, index):
        sample, target = super(LaMemLoader, self).__getitem__(index)
        # Give every sample an alpha of 0
        target = [float(target), 0]

        if self.lamem_target_transform:
            target = self.lamem_target_transform(target)

        return sample, target


class BaseMementoLoader():
    def __init__(self,
                 split,
                 base_path=cfg.MEMENTO_ROOT,
                 load_from='npy',
                 transform=None,
                 target_transform=None):
        assert split in ['train', 'val', 'test']
        assert load_from in ['npy', 'mp4']

        self.split = split
        self.base_path = base_path
        self.load_from = load_from
        self.transform = transform
        self.target_transform = target_transform

        datapath = os.path.join(base_path,
                                "memento_{}_data.json".format(split))
        with open(datapath) as infile:
            self.memento_data = json.load(infile)

    def __len__(self):
        return len(self.memento_data)

    def __getitem__(self, index):
        raise NotImplementedError()

    def load_vid(self, fname):
        load_func = self._load_from_npy if (
            self.load_from == '.npy') else self._load_from_mp4

        return load_func(fname)

    def _load_from_npy(self, fname):
        fpath = os.path.join(self.base_path, "videos_npy",
                             os.path.splitext(fname)[0] + ".npy")

        return np.load(fpath)

    def _load_from_mp4(self, fname):
        fpath = os.path.join(self.base_path, "videos", fname)
        cap = cv2.VideoCaptur(fpath)
        success = True
        frames = []

        while success:
            success, frame = cap.read()

            if success:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return np.array(frames)


class BaseVideoMemLoader():
    def __init__(self):
        pass


class MementoFramesLoader(BaseMementoLoader):
    """
    Loads approximately every 4th frame from the Memento videos.

    Impelements it by loading the npy files and taking every other frame,
    for a total of 23 frames/vid.

    Args:

    Returns:
        x: image frame as a npy array
        y: mem score, alpha for the overall video, as a list
    """
    def __init__(self, nframes=cfg.N_FRAMES_FOR_FRAMES_MODEL, **kwargs):
        super(MementoFramesLoader, self).__init__(load_from='npy', **kwargs)
        self.nframes = nframes

    def __len__(self):
        return len(self.memento_data) * self.nframes

    def _subsample_frames(self, arr):
        step = (len(arr) - 1) / (self.nframes - 1)
        indices = [round(i * step) for i in range(self.nframes)]

        return arr[indices]

    def __getitem__(self, index):
        vid_idx = index // self.nframes
        frame_idx = index % self.nframes
        vid_data = self.memento_data[vid_idx]
        mem_score, alpha = vid_data["mem_score"], vid_data["alpha"]

        # this should give you 45 frames
        frames = self.load_vid(vid_data['filename'])
        frames = self._subsample_frames(frames)
        frame = frames[frame_idx]
        y = [mem_score, alpha]

        if self.transform is not None:
            frame = self.transform(frame)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return frame, y
