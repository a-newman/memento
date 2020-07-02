import os

import torch

import config as cfg


def set_device():
    if cfg.USE_GPU:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda not available")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    return device


def get_ckpt_path(epoch, metrics):
    metrics_str = "__".join(
        ["{}_{}".format(name, val) for name, val in metrics.items()])

    return "ep_{}__{}.pth".format(epoch, metrics_str)


def makedirs(dirs):
    for d in dirs:
        try:
            os.makedirs(d)
        except FileExistsError:
            pass


def log_loss(logger, loss, loss_values, iteration, phase='train'):
    logger.add_scalar('{}Loss'.format(phase.title()), loss.item(), iteration)

    for name, l in loss_values.items():
        logger.add_scalar('{}Loss_{}'.format(phase.title(), name), l.item(),
                          iteration)
