import numpy as np
import torch
import torch.nn as nn


class MemAlphaLoss(nn.Module):
    def __init__(self,
                 device,
                 mse_mem_coeff=100,
                 mse_alpha_coeff=10,
                 decay_curve_coeff=1,
                 npoints=100):
        super(MemAlphaLoss, self).__init__()
        self.mse_mem_coeff = mse_mem_coeff
        self.mse_alpha_coeff = mse_alpha_coeff
        self.decay_curve_coeff = decay_curve_coeff
        self.npoints = npoints
        self.device = device

    def forward(self, y_pred, y_true):
        """Given a vector shape [n], produce vector shape [n, self.npoints]"""
        def duplicate(tensor):
            return tensor[:, None].repeat(1, self.npoints)

        mem_true = y_true[:, 0]
        alpha_true = y_true[:, 1]
        mem_pred = y_pred[:, 0]
        alpha_pred = y_pred[:, 1]

        mse_mem = torch.sum((mem_true - mem_pred)**2)
        mse_alpha = torch.sum((alpha_true - alpha_pred)**2)

        # calculate points along the decay curve
        batch_size = list(y_pred.shape)[0]
        lags = torch.tensor(np.linspace(0, 180, self.npoints))[None, :]
        lags = lags.repeat(batch_size, 1)
        lags = lags.to(self.device)

        mem_true, mem_pred, alpha_true, alpha_pred = map(
            duplicate, [mem_true, mem_pred, alpha_true, alpha_pred])

        decay_curve_true = alpha_true * (lags - 80.) + mem_true
        decay_curve_pred = alpha_pred * (lags - 80.) + mem_pred

        decay_mse = torch.sum((decay_curve_pred - decay_curve_true)**2)

        return (self.mse_mem_coeff *
                mse_mem) + (self.mse_alpha_coeff *
                            mse_alpha) + (self.decay_curve_coeff * decay_mse)
