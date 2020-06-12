import torch.nn as nn
from torchvision.models import densenet121


def get_model(model_name):
    if model_name == "frames":
        return FramesStream()
    else:
        raise RuntimeError("Unrecognized model name {}".format(model_name))


class FramesStream(nn.Module):
    def __init__(self):
        super(FramesStream, self).__init__()
        self.encoder = densenet121(pretrained=True)
        self.fc = nn.Linear(in_features=1000, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.fc(self.encoder(x)))

        return out
