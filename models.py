import torch
import torch.nn as nn
from torchvision.models import densenet121

from kinetics_i3d_pytorch.src.i3dpt import I3D


def get_model(model_name):
    if model_name == "frames":
        return FramesStream()
    elif model_name == "video":
        return VideoStream.get_pretrained()
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


class VideoStream(I3D):
    @classmethod
    def get_pretrained(cls,
                       weights="./kinetics_i3d_pytorch/model/model_rgb.pth",
                       **kwargs):
        model = cls(**kwargs)
        # strict = False will ignore the layers that we don't use
        model.load_state_dict(torch.load(weights), strict=False)

        return model

    def __init__(self):
        super(VideoStream, self).__init__(num_classes=400)
        # weights = "kinetics_i3d_pytorch/model/model_rgb.pth"
        self.final_conv = nn.Conv3d(in_channels=1024,
                                    out_channels=2,
                                    kernel_size=(1, 1, 1),
                                    stride=(1, 1, 1))
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # Preprocessing
        out = self.conv3d_1a_7x7(x)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        # out = self.conv3d_0c_1x1(out)
        out = self.final_conv(out)
        out = out.squeeze(3).squeeze(3).mean(2)
        out = self.activation(out)

        return out
