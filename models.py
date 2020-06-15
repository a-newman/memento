import torch.nn as nn
from torchvision.models import densenet121


def get_model(model_name):
    if model_name == "frames":
        return FramesStream()
    else:
        raise RuntimeError("Unrecognized model name {}".format(model_name))


class FramesModel(nn.Module):
    """Abstract class representing frame-based models."""
    @staticmethod
    def _video_batch_to_frames_batch(x):
        """x is a video batch with dims (batch, c, nframes, h, w)"""
        b_i, c_i, f_i, h_i, w_i = list(range(5))
        b, c, f, h, w = x.shape
        x = x.transpose(f_i, c_i).reshape((b * f, c, h, w))

        return x

    @staticmethod
    def _frames_output_to_video_output(x, frames_per_vid):
        # average the values over the frames
        label_shape = x.shape[1:]
        x = x.reshape((-1, frames_per_vid, *label_shape))
        x = x.mean(dim=1)

        return x

    def image_forward(self, x):
        raise NotImplementedError()

    def forward(self, x):
        # x shape: batch, c, nframes, h, w
        # instead want: batch*nframes, c, h, w

        frames_per_vid = x.shape[2]
        x = self._video_batch_to_frames_batch(x)
        out = self.image_forward(x)

        out = self._frames_output_to_video_output(out, frames_per_vid)

        return out


class FramesStream(FramesModel):
    def __init__(self):
        super(FramesStream, self).__init__()
        self.encoder = densenet121(pretrained=True)
        self.fc = nn.Linear(in_features=1000, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def image_forward(self, x):
        out = self.sigmoid(self.fc(self.encoder(x)))

        return out


class VideoStream(nn.Module):
    def __init__(self):
        super(VideoStream, self).__init__()
        self.encoder = None
        pass

