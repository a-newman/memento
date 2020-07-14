import numpy as np
import torch
import torch.nn as nn

from model_utils import MemCapModelFields, MemModelFields, ModelOutput


class CaptionsLoss(nn.Module):
    def __init__(self, device='cuda', weight=10, class_weights=None):
        super(CaptionsLoss, self).__init__()
        self.device = device
        self.weight = weight

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights,
                                              dtype=torch.float32).to(device)

    def forward(self, y_pred: ModelOutput[MemCapModelFields],
                y_true: ModelOutput[MemCapModelFields]):

        if not ('out_captions' in y_pred and 'out_captions' in y_true):

            return torch.tensor(0, dtype=torch.float32).to(self.device)

        cap_pred = y_pred['out_captions']

        # TODO: we have pre-padded the sequence. Would be more sophisticated
        # to skip the pre-padding and let torch handle the packing for
        # us
        b, t, d = cap_pred.shape
        lens = t * torch.ones(b)
        lens = lens.to(self.device)
        cap_pred_seq = nn.utils.rnn.pack_padded_sequence(cap_pred,
                                                         lens,
                                                         batch_first=True)

        cap_true = y_true['out_captions']
        target = torch.argmax(cap_true, dim=2)
        target_seq = nn.utils.rnn.pack_padded_sequence(target,
                                                       lens,
                                                       batch_first=True)

        args = {'input': cap_pred_seq.data, 'target': target_seq.data}

        if self.class_weights is not None:
            args['weight'] = self.class_weights

        return self.weight * nn.functional.cross_entropy(**args)


class MemMSELoss(nn.Module):
    def __init__(self, device, weights=None):
        super(MemMSELoss, self).__init__()

        if weights is not None:
            self.weights = torch.tensor(weights,
                                        dtype=torch.float32).to(device)
        self.device = device

    def forward(self, y_pred: ModelOutput[MemModelFields],
                y_true: ModelOutput[MemModelFields]):

        mem_true = y_true['score']
        alpha_true = y_true['alpha']
        mem_pred = y_pred['score']
        alpha_pred = y_pred['alpha']

        if self.weights is not None:
            bucket_width = 100 / (len(self.weights) - 1)
            indices = (mem_true / bucket_width).floor().long()
            weights = self.weights[indices]
            mse_mem = nn.functional.mse_loss(mem_pred, mem_true, reduce=None)
            mse_alpha = nn.functional.mse_loss(alpha_pred,
                                               alpha_true,
                                               reduce=None)

            return (weights * mse_mem).mean() + (weights * mse_alpha).mean()
        else:
            mse_mem = nn.functional.mse_loss(mem_pred, mem_true)
            mse_alpha = nn.functional.mse_loss(alpha_pred, alpha_true)

            return mse_mem + mse_alpha


class MemAlphaLoss(nn.Module):
    def __init__(self,
                 device,
                 mse_mem_coeff: float = 1,
                 mse_alpha_coeff: float = .1,
                 decay_curve_coeff: float = .01,
                 npoints: int = 100):
        super(MemAlphaLoss, self).__init__()
        self.mse_mem_coeff = mse_mem_coeff
        self.mse_alpha_coeff = mse_alpha_coeff
        self.decay_curve_coeff = decay_curve_coeff
        self.npoints = npoints
        self.device = device

    def forward(self, y_pred: ModelOutput[MemModelFields],
                y_true: ModelOutput[MemModelFields]):
        """Given a vector shape [n], produce vector shape [n, self.npoints]"""
        def duplicate(tensor):
            return tensor[:, None].repeat(1, self.npoints)

        mem_true = y_true['score']
        alpha_true = y_true['alpha']
        # mem_pred = mem_model_output['score']
        # alpha_pred = mem_model_output['alpha']
        mem_pred = y_pred['score']
        alpha_pred = y_pred['alpha']
        # mem_pred = y_pred[:, 0]
        # alpha_pred = y_pred[:, 1]
        mse_mem = nn.functional.mse_loss(mem_pred, mem_true)
        mse_alpha = nn.functional.mse_loss(alpha_pred, alpha_true)

        # calculate points along the decay curve
        batch_size = list(mem_pred.shape)[0]
        lags = torch.tensor(np.linspace(0, 180, self.npoints))[None, :]
        lags = lags.repeat(batch_size, 1)
        lags = lags.to(self.device)

        mem_true, mem_pred, alpha_true, alpha_pred = map(
            duplicate, [mem_true, mem_pred, alpha_true, alpha_pred])

        decay_curve_true = alpha_true * (lags - 80.) + mem_true
        decay_curve_pred = alpha_pred * (lags - 80.) + mem_pred

        decay_mse = nn.functional.mse_loss(decay_curve_pred, decay_curve_true)

        return (self.mse_mem_coeff *
                mse_mem) + (self.mse_alpha_coeff *
                            mse_alpha) + (self.decay_curve_coeff * decay_mse)
