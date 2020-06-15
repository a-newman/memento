import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision import transforms as T

import config as cfg
from datasets import (MementoMemAlphaLabelSet, MementoRecordSet,
                      VideoRecordLoader)
from torchvideo.samplers import ClipSampler, FrameSampler
from torchvideo.transforms import (CenterCropVideo, CollectFrames,
                                   PILVideoToTensor, RandomCropVideo,
                                   ResizeVideo)

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
VIDEO_TRAIN_TRANSFORMS = T.Compose([
    ResizeVideo((cfg.RESIZE, cfg.RESIZE)),
    RandomCropVideo((cfg.CROP_SIZE, cfg.CROP_SIZE)),
    CollectFrames(),
    PILVideoToTensor()  # TODO Normalize?
])
VIDEO_TEST_TRANSFORMS = T.Compose([
    ResizeVideo((cfg.RESIZE, cfg.RESIZE)),
    CenterCropVideo((cfg.CROP_SIZE, cfg.CROP_SIZE)),
    CollectFrames(),
    PILVideoToTensor()  # TODO Normalize?
])


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
    elif dset_name == "memento_ma":
        train_ds = get_memento_video_loader(split="train",
                                            transform=VIDEO_TRAIN_TRANSFORMS,
                                            target_transform=Y_TRANSFORMS)
        val_ds = get_memento_video_loader(split="val",
                                          transform=VIDEO_TEST_TRANSFORMS,
                                          target_transform=Y_TRANSFORMS)
        test_ds = get_memento_video_loader(split="test",
                                           transform=VIDEO_TEST_TRANSFORMS,
                                           target_transform=Y_TRANSFORMS)
    else:
        raise RuntimeError("Unrecognized dset name: {}".format(dset_name))

    return train_ds, val_ds, test_ds


class NFramesSampler(FrameSampler):
    def __init__(self, nframes):
        self.nframes = nframes

    def sample(self, video_length):
        if video_length == 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".
                format(video_length))

        indices = [
            i * video_length // self.nframes + video_length //
            (2 * self.nframes) for i in range(self.nframes)
        ]

        return indices

    def __repr__(self):
        return self.__class__.__name__ + "(nframes={})".format(self.nframes)


def get_memento_video_loader(split,
                             base_path=cfg.MEMENTO_ROOT,
                             metadata_path=cfg.MEMENTO_METADATA_PATH,
                             transform=None,
                             target_transform=None):
    record_set = MementoRecordSet.from_metadata_file()
    label_set = MementoMemAlphaLabelSet(split=split, base_path=base_path)
    filter_func = lambda r: label_set.is_in_set(r.filename)
    # sampler = ClipSampler(clip_length=45, frame_step=2)
    sampler = NFramesSampler(nframes=45)
    vidloader = VideoRecordLoader(record_set=record_set,
                                  label_set=label_set,
                                  filter=filter_func,
                                  sampler=sampler,
                                  transform=transform,
                                  target_transform=target_transform)

    return vidloader


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


if __name__ == "__main__":
    print("making dataloader")
    # dl = get_memento_video_loader(split="train")
    dl = get_dataset("memento_ma")[0]
    print("done making dataloader")
    print(len(dl))
    x, y = dl[0]
    print("x", x.shape)
    print("y", y)
