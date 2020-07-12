import numpy as np
import torch
import torch.nn as nn

from model_utils import MemCapModelFields, MemModelFields, ModelOutput


class CaptionsLoss(nn.Module):
    def __init__(self, device='cuda', weight=100):
        super(CaptionsLoss, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, y_pred: ModelOutput[MemCapModelFields],
                y_true: ModelOutput[MemCapModelFields]):

        if not (y_pred.get('out_captions', False)
                and y_true.get('out_captions', False)):

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

        return self.weight * nn.functional.cross_entropy(
            input=cap_pred_seq.data, target=target_seq.data)


class MemMSELoss(nn.Module):
    def forward(self, y_pred: ModelOutput[MemModelFields],
                y_true: ModelOutput[MemModelFields]):
        mem_true = y_true['score']
        alpha_true = y_true['alpha']
        mem_pred = y_pred['score']
        alpha_pred = y_pred['alpha']

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
