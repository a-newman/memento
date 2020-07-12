import json
import os

import numpy as np
from PIL import Image
from torchvideo.datasets import LabelSet, VideoFolderDataset
from torchvideo.transforms import Transform

import config as cfg

class ChannelLast(Transform):
    def _gen_params(self, frames):
        return None

    def _transform(self, frames, params):
        # c, nframes, h, w
        # instead make it: frames, h, w, c
        tensor_ndim = len(frames.size())

        if tensor_ndim != 4:
            raise ValueError("Expected 4d tensor, but got shape {}".format(
                frames.shape))

        frames = frames.permute(1, 2, 3, 0)

        return frames


class MementoLabelSet(LabelSet):
    def __init__(self, split, base_path, metadata_path):
        self.split = split
        self.base_path = base_path

        datapath = os.path.join(base_path,
                                "memento_{}_data.json".format(split))
        with open(datapath) as infile:
            memento_data = json.load(infile)

        self.memento_data = {
            os.path.splitext(elt['filename'])[0]: elt
            for elt in memento_data
        }

        with open(metadata_path) as infile:
            self.metadata = json.load(infile)

    def __getitem__(self, vidname):
        viddata = self.memento_data[os.path.splitext(vidname)[0]]

        return [viddata['mem_score'], viddata['alpha']]

    def is_in_set(self, vidpath):
        vidname = os.path.splitext(os.path.basename(vidpath))[0]

        return vidname in self.memento_data

    def num_frames(self, vidpath):
        vidname = os.path.splitext(os.path.basename(vidpath))[0]

        # return self.metadata[vidname]['n_frames']

        return 45


def _is_video_file(path):
    ext = path.name.lower().split(".")[-1]

    return ext in [
        "mp4", "webm", "avi", "3gp", "wmv", "mpg", "mpeg", "mov", "mkv", "npy"
    ]


class VideoOrNpyFolderDataset(VideoFolderDataset):
    def _load_frames(self, frames_idx, video_file):
        from torchvideo.internal.readers import default_loader
        from torchvideo.samplers import frame_idx_to_list

        if os.path.splitext(video_file)[-1] in ['.NPY', '.npy']:
            vid = np.load(video_file)
            frames_idx = np.array(frame_idx_to_list(frames_idx))
            vid = vid[frames_idx]

            return (Image.fromarray(frame) for frame in vid)
        else:
            return default_loader(video_file, frames_idx)

    @staticmethod
    def _get_video_paths(root_path, filter):
        return sorted([
            child for child in root_path.iterdir()
            if _is_video_file(child) and (filter is None or filter(child))
        ])


def get_memento_video_loader(split,
                             base_path=cfg.MEMENTO_ROOT,
                             metadata_path=cfg.MEMENTO_METADATA_PATH,
                             transform=None):
    memento_vid_path = os.path.join(base_path, "videos_npy")
    label_set = MementoLabelSet(split=split,
                                base_path=base_path,
                                metadata_path=metadata_path)
    filter = label_set.is_in_set
    frame_counter = label_set.num_frames
    # sampler = None
    vidloader = VideoOrNpyFolderDataset(
        root_path=memento_vid_path,
        label_set=label_set,
        # sampler=sampler,
        filter=filter,
        frame_counter=frame_counter,
        transform=transform)

    return vidloader


class MementoFramesLoader():
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
