import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision import transforms as T

import config as cfg
from datasets import (MementoMemAlphaCapLabelSet, MementoMemAlphaLabelSet,
                      MementoRecordSet, VideoRecordLoader)
from model_utils import ModelOutput
from torchvideo.samplers import ClipSampler, FrameSampler, FullVideoSampler
from torchvideo.transforms import (CenterCropVideo, CollectFrames,
                                   PILVideoToTensor, RandomCropVideo,
                                   ResizeVideo, TimeToChannel, Transform)


class RescaleInRange(Transform):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        assert self.lower < self.upper

    def _gen_params(self, frames):
        return None

    def _transform(self, frames, params):
        maxval = torch.max(frames)
        minval = torch.min(frames)
        spread = self.upper - self.lower
        current_spread = maxval - minval

        return (frames - minval) * (spread / current_spread) + self.lower


class ApplyToKeys(object):
    """Applies transforms to the keys of a ModelOutput obj"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample: ModelOutput):
        data = {k: self.transform([v])[0] for k, v in sample.items()}

        return ModelOutput(data)


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
Y_TRANSFORMS = ApplyToKeys(torch.FloatTensor)
# Y_TRANSFORMS = None
VIDEO_TRAIN_TRANSFORMS = T.Compose([
    ResizeVideo(cfg.RESIZE),
    RandomCropVideo(cfg.CROP_SIZE),
    CollectFrames(),
    PILVideoToTensor(rescale=True),
    RescaleInRange(-1, 1)
])
VIDEO_TEST_TRANSFORMS = T.Compose([
    ResizeVideo((cfg.RESIZE, cfg.RESIZE)),
    CenterCropVideo((cfg.CROP_SIZE, cfg.CROP_SIZE)),
    CollectFrames(),
    PILVideoToTensor(rescale=True),
    RescaleInRange(-1, 1)
])
FRAMES_TRAIN_TRANSFORMS = T.Compose([VIDEO_TRAIN_TRANSFORMS, TimeToChannel()])
FRAMES_TEST_TRANSFORMS = T.Compose([VIDEO_TEST_TRANSFORMS, TimeToChannel()])


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
        train_ds = memento_frames_loader(split="train",
                                         transform=VIDEO_TRAIN_TRANSFORMS,
                                         target_transform=Y_TRANSFORMS)
        val_ds = memento_frames_loader(split="val",
                                       transform=VIDEO_TEST_TRANSFORMS,
                                       target_transform=Y_TRANSFORMS)
        test_ds = memento_frames_loader(split="test",
                                        transform=VIDEO_TEST_TRANSFORMS,
                                        target_transform=Y_TRANSFORMS)
    elif (dset_name == "memento_ma") or (dset_name == "memento_ma_cap"):
        with_captions = dset_name == "memento_ma_cap"
        y_transform = Y_TRANSFORMS if not with_captions else None
        print("Y TRANSFORM", y_transform)
        train_ds = memento_video_loader(split="train",
                                        transform=VIDEO_TRAIN_TRANSFORMS,
                                        target_transform=y_transform,
                                        with_captions=with_captions)
        val_ds = memento_video_loader(split="val",
                                      transform=VIDEO_TEST_TRANSFORMS,
                                      target_transform=y_transform,
                                      with_captions=with_captions)
        test_ds = memento_video_loader(split="test",
                                       transform=VIDEO_TEST_TRANSFORMS,
                                       target_transform=y_transform,
                                       with_captions=with_captions)
    else:
        raise RuntimeError("Unrecognized dset name: {}".format(dset_name))

    return train_ds, val_ds, test_ds


class NRandomFramesSampler(FrameSampler):
    def __init__(self, nframes):
        self.nframes = nframes

    def sample(self, video_length):
        if video_length < 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".
                format(video_length))
        indices = []

        while len(indices) < self.nframes:
            indices.extend(list(range(video_length - 1)))

        indices = sorted(random.sample(indices, k=self.nframes))

        return indices


class NFramesSampler(FrameSampler):
    def __init__(self, nframes, avoid_final_frame=True):
        self.nframes = nframes
        self.avoid_final_frame = avoid_final_frame

    def sample(self, video_length):
        if self.avoid_final_frame:
            video_length = video_length - 1

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


def ComposedSampler(FrameSampler):
    def __init__(self, samplers):
        assert len(samplers) > 0
        self.samplers = samplers

    def sample(self, video_length):
        indices = np.array(list(range(video_length)))

        for sampler in self.samplers:
            sampler_indices = sampler.sample(len(indices))
            indices = indices[sampler_indices]

        return indices

    def __repr__(self):
        return self.__class__.__name__ + "(samplers={})".format(self.samplers)


def memento_frames_loader(split, transform, target_transform, nframes=23):
    # will select the center frame

    if split == "train":
        # sampler = NRandomFramesSampler(nframes=nframes)
        sampler = NRandomFramesSampler(nframes=nframes)
        # sampler = FullVideoSampler()
    else:
        sampler = NFramesSampler(nframes=nframes)

    return get_memento_video_loader(split,
                                    sampler,
                                    transform=transform,
                                    target_transform=target_transform)


def memento_video_loader(split,
                         transform,
                         target_transform,
                         with_captions=False):
    sampler = NFramesSampler(nframes=45)

    return get_memento_video_loader(split,
                                    sampler,
                                    transform=transform,
                                    target_transform=target_transform,
                                    with_captions=with_captions)


def get_memento_video_loader(split,
                             sampler,
                             metadata_path=cfg.MEMENTO_METADATA_PATH,
                             transform=None,
                             target_transform=None,
                             with_captions=False):
    record_set = MementoRecordSet.from_metadata_file()

    if with_captions:
        label_set = MementoMemAlphaCapLabelSet(split=split, factor=100)
    else:
        label_set = MementoMemAlphaLabelSet(split=split, factor=100)
    filter_func = lambda r: label_set.is_in_set(r.filename)
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

    dl = get_dataset("memento_frames")[0]

    for i in range(100):
        x, y = dl[i]
        print(x.shape)
