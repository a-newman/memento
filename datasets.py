import json
import os

from torchvideo.datasets import LabelSet, VideoDataset
from torchvideo.internal.readers import default_loader
from torchvideo.samplers import _default_sampler
from torchvideo.transforms import PILVideoToTensor

import config as cfg


class VideoRecord:
    """
    Represents a video record.
    """
    def __init__(self, data):
        self.data = data

    @property
    def path(self):
        return self.data['path']

    @property
    def filename(self):
        return self.data['filename']

    @property
    def num_frames(self):
        return self.data.get('num_frames')

    @property
    def height(self):
        return self.data.get('height')

    @property
    def width(self):
        return self.data.get('width')

    @property
    def size(self):
        return (self.height, self.width)

    @property
    def label(self):
        return self.data['label']

    @property
    def category(self):
        return self.data['category']

    def todict(self):
        return self.data

    def __hash__(self):
        return hash(self.data.values())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.path == other.path
        else:
            return False

    def __repr__(self):
        return str(self.todict())


class VideoRecordSet:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]

    @classmethod
    def from_metadata_file(cls, fpath):
        with open(fpath) as infile:
            data = json.load(infile)
        records = [VideoRecord(elt) for elt in data]

        return cls(records=records)


class MementoRecordSet(VideoRecordSet):
    @classmethod
    def from_metadata_file(cls):
        fpath = cfg.MEMENTO_METADATA_PATH

        return VideoRecordSet.from_metadata_file(fpath)


class VideoRecordLoader(VideoDataset):
    def __init__(self,
                 record_set,
                 filter=None,
                 label_set=None,
                 sampler=_default_sampler(),
                 loader=default_loader,
                 transform=PILVideoToTensor(),
                 target_transform=int):
        self.filter = filter
        self.record_set = record_set if filter is None else [
            r for r in record_set if filter(r)
        ]
        self.label_set = label_set
        self.labels = self._label_examples(self.record_set, self.label_set)
        self.sampler = sampler
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def _label_examples(record_set, label_set):
        if label_set:
            return [label_set[record.path] for record in record_set]
        else:
            return [record.label for record in record_set]

    def _load_frames(self, vidpath, indices):
        return self.loader(vidpath, indices)

    def __len__(self):
        return len(self.record_set)

    def __getitem__(self, index):
        record = self.record_set[index]
        vidpath = record.path
        vidlen = record.num_frames
        frame_indices = self.sampler.sample(vidlen)
        frames = self._load_frames(vidpath, frame_indices)
        label = self.labels[index]

        if self.transform is not None:
            frames = self.transform(frames)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label


class MementoLabelSet(LabelSet):
    def __init__(self, split, base_path):
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

    def data_for_vidpath(self, vidpath):
        return self.memento_data[os.path.splitext(
            os.path.basename(vidpath))[0]]

    def is_in_set(self, vidname):
        return vidname in self.memento_data


class MementoMemAlphaLabelSet(MementoLabelSet):
    def __getitem__(self, vidpath):
        viddata = self.data_for_vidpath(vidpath)

        return [viddata['mem_score'], viddata['alpha']]
