import os
import time

import fire
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from data_loader import get_dataset
from losses import MemAlphaLoss
from models import get_model


def save_ckpt(savepath, model, epoch, it, optimizer, dataset_name, model_type):
    torch.save(
        {
            'epoch': epoch,
            'it': it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataset_name': dataset_name,
            'model_type': model_type
        }, savepath)


def main(verbose=1,
         print_freq=100,
         restore=True,
         ckpt_path=None,
         val_freq=1,
         run_id="model",
         dset_name="memento_frames",
         model_name="frames",
         freeze_encoder_until_it=1000):

    print("TRAINING MODEL {} ON DATASET {}".format(model_name, dset_name))

    if restore and ckpt_path:
        raise RuntimeError("Specify restore 0R ckpt_path")

    ckpt_savepath = os.path.join(cfg.CKPT_DIR, "{}.pth".format(run_id))
    print("Saving ckpts to {}".format(ckpt_savepath))
    logs_savepath = os.path.join(cfg.LOGDIR, run_id)
    print("Saving logs to {}".format(logs_savepath))

    if restore or ckpt_path:
        print("Restoring weights from {}".format(
            ckpt_savepath if restore else ckpt_path))

    if cfg.USE_GPU:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda not available")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    print('DEVICE', device)

    # model
    model = get_model(model_name)
    model = DataParallel(model)

    # must call this before constructing the optimizer:
    # https://pytorch.org/docs/stable/optim.html
    model.to(device)

    # set up training
    # TODO better one?
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    criterion = MemAlphaLoss(device=device)

    initial_epoch = 0
    iteration = 0
    unfrozen = False

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['model_state_dict']

        model.load_state_dict(state_dict)

    elif restore:
        if os.path.exists(ckpt_savepath):
            print("LOADING MODEL")
            ckpt = torch.load(ckpt_savepath)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            initial_epoch = ckpt['epoch']
            iteration = ckpt['it']

    else:
        raise RuntimeError("Should not get here! Check for bugs")

    # dataset
    train_ds, val_ds, test_ds = get_dataset(dset_name)
    assert val_ds or test_ds

    # train_ds = Subset(train_ds, range(500))
    # test_ds = Subset(test_ds, range(100))
    train_dl = DataLoader(train_ds,
                          batch_size=cfg.BATCH_SIZE,
                          shuffle=True,
                          num_workers=cfg.NUM_WORKERS)
    test_dl = DataLoader(test_ds,
                         batch_size=cfg.BATCH_SIZE,
                         shuffle=False,
                         num_workers=cfg.NUM_WORKERS)

    # training loop
    start = time.time()

    try:
        for epoch in range(initial_epoch, cfg.NUM_EPOCHS):
            logger = SummaryWriter(logs_savepath)

            # effectively puts the model in train mode.
            # Opposite of model.eval()
            model.train()

            print("Epoch {}".format(epoch))

            for i, (x, y) in tqdm(enumerate(train_dl),
                                  total=len(train_ds) / cfg.BATCH_SIZE):
                iteration += 1

                if not unfrozen and iteration > freeze_encoder_until_it:
                    print("Unfreezing encoder")
                    unfrozen = True

                    for param in model.parameters():
                        param.requires_grad = True

                logger.add_scalar('DataTime', time.time() - start, iteration)

                x = x.to(device)
                y = y.to(device)

                out = model(x)
                loss = criterion(out, y)

                # I think this zeros out previous gradients (in case people
                # want to accumulate gradients?)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logging
                logger.add_scalar('TrainLoss', loss.item(), iteration)
                logger.add_scalar('ItTime', time.time() - start, iteration)
                start = time.time()

                # display metrics

            # do some validation

            if (epoch + 1) % val_freq == 0:
                print("Validating...")
                model.eval()  # puts model in validation mode

                with torch.no_grad():

                    for i, (x, y) in tqdm(enumerate(test_dl),
                                          total=len(test_ds) / cfg.BATCH_SIZE):
                        x = x.to(device)
                        y = y.to(device)

                        out = model(x)
                        loss = criterion(x, y)

                        logger.add_scalar('ValLoss', loss, iteration)

            # end of epoch
            lr_scheduler.step()

            save_ckpt(ckpt_savepath, model, epoch, iteration, optimizer,
                      dset_name, model_name)

    except KeyboardInterrupt:
        print('Got keyboard interrupt, saving model...')
        save_ckpt(ckpt_savepath, model, epoch, iteration, optimizer, dset_name,
                  model_name)


if __name__ == "__main__":
    fire.Fire(main)
