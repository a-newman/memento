import json
import os
import random

import numpy as np

import config as cfg
from cap_utils import transform_caption
from model_utils import MemCapModelFields, MemModelFields, ModelOutput
from torchvideo.datasets import LabelSet, VideoDataset
from torchvideo.internal.readers import default_loader
from torchvideo.samplers import _default_sampler
from torchvideo.transforms import PILVideoToTensor


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
                 target_transform=int,
                 preload_labels=False):
        self.filter = filter
        self.record_set = record_set if filter is None else [
            r for r in record_set if filter(r)
        ]
        self.label_set = label_set
        self.preload_labels = preload_labels

        if self.preload_labels:
            self.labels = self._label_examples(self.record_set, self.label_set)
        self.sampler = sampler
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def get_fnames(self):
        return [r.filename for r in self.record_set]

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

        if self.preload_labels:
            label = self.labels[index]
        else:
            label = self.label_set[record.path]

        if self.transform is not None:
            frames = self.transform(frames)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label


class MementoLabelSet(LabelSet):
    def __init__(self, split, base_path=cfg.MEMENTO_ROOT):
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
        return self.memento_data[self.vidname_from_path(vidpath)]

    def vidname_from_path(self, vidpath):
        return os.path.splitext(os.path.basename(vidpath))[0]

    def is_in_set(self, vidname):
        return vidname in self.memento_data


class MementoMemAlphaLabelSet(MementoLabelSet):
    def __init__(self, split, factor=100):
        MementoLabelSet.__init__(self, split)
        self.factor = factor

    def __getitem__(self, vidpath) -> ModelOutput[MemModelFields]:
        viddata = self.data_for_vidpath(vidpath)

        mem_score = self.factor * viddata['mem_score']
        alpha = self.factor * viddata['alpha']

        return ModelOutput({'score': mem_score, 'alpha': alpha})


class MementoMemAlphaCapLabelSet(MementoLabelSet):
    def __init__(self, split, factor=100):
        MementoLabelSet.__init__(self, split)
        self.factor = factor

        with open(cfg.MEMENTO_CAPTIONS_EMBEDDING) as infile:
            self.word_embedding = json.load(infile)

        caps_path = cfg.MEMENTO_CAPTIONS_DATA
        with open(caps_path) as infile:
            self.cap_data = json.load(infile)

    def get_full_cap_data(self, vidpath):
        cap_data = self.cap_data[self.vidname_from_path(vidpath)]
        cap_i = random.randint(0, len(cap_data['indexed_captions']) - 1)
        cap, tokenized_cap = cap_data['indexed_captions'][cap_i], cap_data[
            'tokenized_captions'][cap_i]

        cap_in, cap_out = transform_caption(
            cap,
            tokenized_cap,
            input_format="embedding_list",
            caption_format="one_hot_list",
            add_padding=True,
            word_embeddings=self.word_embedding,
            max_cap_len=cfg.MAX_CAP_LEN,
            vocab_size=cfg.VOCAB_SIZE)

        return {
            'in': cap_in,
            'out': cap_out,
            'indexed': cap,
            'tokenized': tokenized_cap
        }

    def __getitem__(self, vidpath) -> ModelOutput[MemCapModelFields]:
        viddata = self.data_for_vidpath(vidpath)
        score, alpha = self.factor * viddata[
            'mem_score'], self.factor * viddata['alpha']

        cap_data = self.cap_data[self.vidname_from_path(vidpath)]
        cap_i = random.randint(0, len(cap_data['indexed_captions']) - 1)
        cap, tokenized_cap = cap_data['indexed_captions'][cap_i], cap_data[
            'tokenized_captions'][cap_i]

        cap_in, cap_out = transform_caption(
            cap,
            tokenized_cap,
            input_format="embedding_list",
            caption_format="one_hot_list",
            add_padding=True,
            word_embeddings=self.word_embedding,
            max_cap_len=cfg.MAX_CAP_LEN,
            vocab_size=cfg.VOCAB_SIZE)

        return ModelOutput({
            'score':
            score,
            'alpha':
            alpha,
            'in_captions':
            cap_in.astype(np.float32, copy=False),
            'out_captions':
            cap_out.astype(np.float32, copy=False),
        })
