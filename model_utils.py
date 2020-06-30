from collections import abc
from typing import Any, Generic, List, Mapping, TypedDict, TypeVar, Union

import torch

T = TypeVar('T', bound=Mapping)


class ModelOutput(abc.Mapping, Generic[T]):
    def __init__(self, data: T):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self.data.__iter__()

    def __repr__(self):
        return "{}(data={})".format(self.__class__.__name__, str(self.data))

    @classmethod
    def _factory(cls, data: T):
        return cls(data=data)

    def to_numpy(self) -> 'ModelOutput[T]':
        data: T = {k: v.numpy() for k, v in self.data.items()}  # type: ignore

        return self._factory(data)

    def to_device(self, device) -> 'ModelOutput[T]':
        data: T = {
            k: v.to(device)  # type: ignore
            for k, v in self.data.items()
        }

        return self._factory(data)

    def merge(self, other: 'ModelOutput[T]'):
        for k in self.data.keys():
            self.data[k].extend(other.data[k])


class MemModelFields(TypedDict):
    score: Union[float, torch.Tensor]
    alpha: Union[float, torch.Tensor]


class MemCapModelFields(MemModelFields):
    in_captions: Union[List[float], torch.Tensor]
    out_captions: Union[List[float], torch.Tensor]


# class MemModelOutput(ModelOutput):
#     def __init__(self, data: MemModelFields):
#         self.data = data

if __name__ == "__main__":
    fields: MemModelFields = {'score': 1, 'alpha': -0.0001}
    out: ModelOutput[MemModelFields] = ModelOutput({
        'score': 1,
        'alpha': -0.0001,
    })

    nump = out.to_device('blah')
    reveal_type(nump)
