import torch
import torch.nn as nn
from torchvision.models import densenet121

import config as cfg
from kinetics_i3d_pytorch.src.i3dpt import I3D
from model_utils import MemCapModelFields, MemModelFields, ModelOutput


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
                 n_hidden_units=512,
                 freeze_encoder=True):
        super(VideoStreamLSTM, self).__init__()
        self.max_caption_size = max_caption_size
        self.vocab_size = vocab_size
        self.feature_dim = 1024
        self.n_hidden_units = n_hidden_units
        self.lstm_input_size = 300
        self.dropout = 0.5
        self.device = device
        self.freeze_encoder = freeze_encoder

        self.base = HeadlessI3D.get_pretrained()

        # mem alpha branch
        self.final_conv = self._unit_conv(in_dim=self.feature_dim, out_dim=2)
        self.activation = nn.LeakyReLU()

        # captions branch
        self.init_h = self._unit_conv(in_dim=self.feature_dim,
                                      out_dim=self.n_hidden_units)
        self.init_c = self._unit_conv(in_dim=self.feature_dim,
                                      out_dim=self.n_hidden_units)
        # self.lstm = nn.LSTM(input_size=self.lstm_input_size,
        #                     hidden_size=self.n_hidden_units,
        #                     num_layers=2)

        # input_size must be the size of the input word vectors, i.e. size of
        # the fasttext embedding
        self.lstm_step = nn.LSTMCell(input_size=self.lstm_input_size,
                                     hidden_size=self.n_hidden_units)
        self.cap_fc = nn.Linear(self.n_hidden_units, self.vocab_size)
        self.cap_dropout = nn.Dropout(p=self.dropout)
        self.cap_activation = nn.Softmax(dim=1)

        if self.freeze_encoder:
            for param in self.base.parameters():
                param.requires_grad = False

    @staticmethod
    def _unit_conv(in_dim, out_dim):
        return nn.Conv3d(in_channels=in_dim,
                         out_channels=out_dim,
                         kernel_size=(1, 1, 1),
                         stride=(1, 1, 1))

    def encode(self, x):
        return self.base(x)

    def decode(self, features):
        pass

    def init_hidden_state(self, features):
        h = self.init_h(features).squeeze(3).squeeze(3).mean(2)
        # print("h shape", h.shape)
        c = self.init_c(features).squeeze(3).squeeze(3).mean(2)
        # cap dim seq_len, batch, input_size

        return h, c

    def caption_decode_step(self, h, c, inputs):
        newh, newc = self.lstm_step(inputs, (h, c))
        logits = self.cap_fc(self.cap_dropout(h))
        preds = self.cap_activation(logits)

        return newh, newc, preds

    def forward(self, x,
                label: ModelOutput[MemCapModelFields]) -> MemCapModelFields:

        # print("x shape", x.shape)
        cap_inp = label['in_captions']
        # print("cap inp shape", cap_inp.shape)

        features = self.encode(x)
        # print("features shape", features.shape)
        batch_size = features.size(0)

        # mem-alpha branch
        mem_out = self.final_conv(features)
        mem_out = mem_out.squeeze(3).squeeze(3).mean(2)
        mem_out = self.activation(mem_out)
        mem_scores = mem_out[:, 0]
        alphas = mem_out[:, 1]

        # captions branch

        h, c = self.init_hidden_state(features)

        predictions = torch.zeros(batch_size, self.max_caption_size,
                                  self.vocab_size).to(self.device)

        # cap inp: batch x max_cap_len x vocab_embed_size
        # = (b x 50 x 300)

        for i in range(self.max_caption_size):
            inp = cap_inp[:, i, :]  # (batch size, input size)
            h, c, preds = self.caption_decode_step(h, c, inp)
            # print("inp", inp.shape)
            predictions[:, i, :] = preds

        data: MemCapModelFields = {
            'score': mem_scores,
            'alpha': alphas,
            'in_captions': cap_inp,
            'out_captions': predictions
        }

        return data


class VideoStream(nn.Module):
    def __init__(self):
        super(VideoStream, self).__init__()
        self.base = HeadlessI3D.get_pretrained()
        self.final_conv = nn.Conv3d(in_channels=1024,
                                    out_channels=2,
                                    kernel_size=(1, 1, 1),
                                    stride=(1, 1, 1))
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, *_) -> MemModelFields:
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
