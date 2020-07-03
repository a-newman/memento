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


def try_load_state_dict(model, state_dict, require_use_strict):
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as re:
        if require_use_strict:
            raise (re)
        else:
            print("WARNING! Error loading model with strict set")
            print(re)
            print("Trying with strict not set...")
            model.load_state_dict(state_dict, strict=False)
            print("Ok.")


def try_load_optim_state(optimizer, state_dict, require_strict):
    try:
        optimizer.load_state_dict(state_dict)
    except ValueError as ve:
        if require_strict:
            raise ve
        else:
            print("WARNING! Unable to restore the optimizer state dict")
            print(ve)
            print("Skipping.")
