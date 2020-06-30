import torch
import torch.nn as nn
from torchvision.models import densenet121

import config as cfg
from kinetics_i3d_pytorch.src.i3dpt import I3D
from model_utils import MemModelFields, ModelOutput


def get_model(model_name, device):
    if model_name == "frames":
        return FramesStream()
    elif model_name == "video":
        return VideoStream()
    elif model_name == "video_lstm":
        return VideoStreamLSTM(device)
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


class VideoStreamLSTM(nn.Module):
    """ IN PROGRESS
    See https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py
    for implementation of show, attend, and tell

    """
    def __init__(self,
                 device,
                 max_caption_size=cfg.MAX_CAP_LEN,
                 vocab_size=cfg.VOCAB_SIZE,
                 n_hidden_units=512):
        super(VideoStreamLSTM, self).__init__()
        self.max_caption_size = max_caption_size
        self.vocab_size = vocab_size
        self.feature_dim = 1024
        self.n_hidden_units = n_hidden_units
        self.lstm_input_size = 300
        self.dropout = 0.5
        self.device = device

        self.base = HeadlessI3D.get_pretrained()

        # mem alpha branch
        self.final_conv = self._unit_conv(in_dim=self.n_hidden_units,
                                          out_dim=2)
        self.activation = nn.LeakyReLU()

        # captions branch
        self.init_h = self._unit_conv(in_dim=self.n_hidden_units,
                                      out_dim=self.n_hidden_units)
        self.init_c = self._unit_conv(in_dim=self.n_hidden_units,
                                      out_dim=self.n_hidden_units)
        # self.lstm = nn.LSTM(input_size=self.lstm_input_size,
        #                     hidden_size=self.n_hidden_units,
        #                     num_layers=2)
        self.lstm_step = nn.LSTMCell()
        # self.post_lstm_fc = nn.Linear(in_features=None,
        #                               out_features=self.vocab_size)
        self.cap_fc = nn.Linear(self.n_hidden_units, self.vocab_size)
        self.cap_dropout = nn.Dropout(p=self.dropout)
        self.cap_activation = nn.Softmax()
        self.lstm_activation = nn.SoftMax()

    @staticmethod
    def _unit_conv(in_dim, out_dim):
        return nn.Conv3d(in_channels=in_dim,
                         out_channels=out_dim,
                         kernel_size=(1, 1, 1),
                         stride=(1, 1, 1))

    def forward(self, x, cap_inp):
        features = self.base(x)
        batch_size = features.size(0)

        # mem-alpha branch
        mem_out = self.final_conv(features)
        mem_out = mem_out.squeeze(3).squeeze(3).mean(2)
        mem_out = self.activation(mem_out)

        # captions branch
        h = self.init_h0(features)
        c = self.init_c0(features)
        # cap dim seq_len, batch, input_size
        # cap_inp = cap
        # cap_out, (hn, cn) = self.lstm(cap_inp, (h0, c0))

        predictions = torch.zeros(batch_size, self.max_caption_size,
                                  self.vocab_size).to(self.device)

        for i in range(self.max_caption_size):
            inp = cap_inp[i]  # batch size, input size
            h, c = self.lstm_step(inp, (h, c))
            preds = self.cap_activation(self.cap_fc(self.cap_dropout(h)))
            predictions[:, i, :] = preds

        mem_scores = mem_out[:, 0]
        alphas = mem_out[:, 1]
        captions = predictions

        return model_utils.MemModelCaptionsOutput.pred(mem_score=mem_scores,
                                                       captions=captions,
                                                       alpha=alphas)


class VideoStream(nn.Module):
    def __init__(self):
        super(VideoStream, self).__init__()
        self.base = HeadlessI3D.get_pretrained()
        self.final_conv = nn.Conv3d(in_channels=1024,
                                    out_channels=2,
                                    kernel_size=(1, 1, 1),
                                    stride=(1, 1, 1))
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> MemModelFields:
        out = self.base(x)
        out = self.final_conv(out)
        out = out.squeeze(3).squeeze(3).mean(2)
        out = self.activation(out)

        mem_scores = out[:, 0]
        alphas = out[:, 1]

        data: MemModelFields = {'score': mem_scores, 'alpha': alphas}

        return data


class HeadlessI3D(I3D):
    @classmethod
    def get_pretrained(cls,
                       weights="./kinetics_i3d_pytorch/model/model_rgb.pth",
                       **kwargs):
        model = cls(num_classes=400, **kwargs)
        # strict = False will ignore the layers that we don't use
        model.load_state_dict(torch.load(weights), strict=False)

        return model

    def forward(self, inp):
        out = self.conv3d_1a_7x7(inp)
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
        # out = out.squeeze(3)
        # out = out.squeeze(3)
        # out = out.mean(2)
        # out_logits = out
        # out = self.softmax(out_logits)

        return out
