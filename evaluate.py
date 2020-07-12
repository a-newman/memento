import json
import os
from typing import Callable, Mapping, Optional

import fire
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import cap_utils
import config as cfg
import utils
from data_loader import get_dataset
from model_utils import MemModelFields, ModelOutput
from models import VideoStreamLSTM, get_model


def rc(labels: ModelOutput[MemModelFields],
       preds: ModelOutput[MemModelFields],
       _,
       verbose=True):
    mem_scores_true = labels['score']
    mem_scores_pred = preds['score']
    val = spearmanr(mem_scores_true, mem_scores_pred)

    return val.correlation


def predict(ckpt_path,
            metrics: Mapping[str, Callable] = {'rc': rc},
            num_workers: int = 20,
            use_gpu: bool = True,
            model_name: str = "frames",
            dset_name: str = "memento_frames",
            batch_size: int = 1,
            preds_savepath: Optional[str] = None,
            use_val: bool = False,
            debug_n: Optional[int] = None,
            should_predict_captions: bool = False):

    print("ckpt path: {}".format(ckpt_path))

    if preds_savepath is None:
        preds_savepath = os.path.splitext(
            ckpt_path.replace(cfg.CKPT_DIR, cfg.PREDS_DIR))[0] + '.json'
        utils.makedirs([os.path.dirname(preds_savepath)])
    print("preds savepath: {}".format(preds_savepath))

    device = utils.set_device()
    print('DEVICE', device)

    # load the vocab embedding
    with open(cfg.MEMENTO_CAPTIONS_EMBEDDING) as infile:
        vocab_embedding = json.load(infile)

    # load the vocab itself
    word2idx, idx2word = cap_utils.index_vocab()

    # load the ckpt
    print("Loading model from path: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path)

    # model
    model = get_model(model_name, device)
    model = nn.DataParallel(model)
    model.load_state_dict(ckpt['model_state_dict'])

    model.to(device)
    model.eval()

    print("USING MODEL TYPE {} ON DSET {}".format(model_name, dset_name))

    # data loader
    train, val, test = get_dataset(dset_name)
    ds = val if use_val else test

    if ds is None:
        raise ValueError("No {} set available for this dataset.".format(
            "val" if use_val else "test"))

    if debug_n is not None:
        ds = Subset(ds, range(debug_n))

    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)

    preds: Optional[ModelOutput] = None
    labels: Optional[ModelOutput] = None
    with torch.no_grad():
        for i, (x, y_) in tqdm(enumerate(dl), total=len(ds) / batch_size):

            y: ModelOutput[MemModelFields] = ModelOutput(y_)
            y_list = y.to_numpy()
            labels = y_list if labels is None else labels.merge(y_list)

            x = x.to(device)
            y = y.to_device(device)

            if should_predict_captions:
                assert batch_size == 1  # required for beam search
                words = predict_captions_simple(model, x, device,
                                                vocab_embedding, idx2word)
                print(words)
                assert False

            out = ModelOutput(model(x, y.get_data()))

            out_list = out.to_device('cpu').to_numpy()
            preds = out_list if preds is None else preds.merge(out_list)

    metrics = {fname: f(labels, preds, None) for fname, f in metrics.items()}
    print("METRICS", metrics)

    data = {
        'ckpt': ckpt_path,
        'preds': preds.to_list().get_data(),
        'labels': labels.to_list().get_data(),
        'metrics': metrics
    }

    with open(preds_savepath, "w") as outfile:
        print("Saving results")
        json.dump(data, outfile)


def predict_captions_simple(model, x, device, vocab_embedding, idx2word):
    start_token_embedded = vocab_embedding['<start>']
    inp = torch.Tensor([start_token_embedded])
    inp = inp.to(device)

    features = model.module.encode(x)  # (batch(1), 1024, 5, 1, 1)

    # initialize the lstm
    h, c = model.module.init_hidden_state(features)

    words = []

    for step in range(cfg.MAX_CAP_LEN):
        h, c, out = model.module.caption_decode_step(h, c, inp)
        out_numpy = out.to("cpu").numpy()
        token = cap_utils.one_hot_to_token(out_numpy, idx2word)
        inp = torch.Tensor([vocab_embedding[token]]).to(device)
        print("preds shape", out.shape)
        words.append(token)

    return words


def predict_captions_beam(model,
                          x: torch.Tensor,
                          device,
                          vocab_embedding,
                          beam_size: int = 3) -> None:
    """IN PROGRESS"""
    k = beam_size
    # seed your list of captions;
    start_token_embedded = vocab_embedding['<start>']
    # dim k x emb_len
    prev_words = torch.Tensor([start_token_embedded] * k)
    prev_words = prev_words.to(device)

    # Setup #################################################

    # holds the scores for top k beam search contenders
    top_k_scores = torch.zeros(k, 1).to(device)

    # Running the search ####################################

    # encode the image
    features = model.module.encode(x)  # (batch(1), 1024, 5, 1, 1)

    # duplicate the features k times
    newshape = [k] + list(features.shape)[1:]
    features = features.expand(newshape)  # (k, 1024, 5, 1, 1)

    # initialize the lstm
    h, c = model.module.init_hidden_state(features)

    for step in range(cfg.MAX_CAP_LEN):
        h, c, preds = model.module.caption_decode_step(h, c, prev_words)
        print("preds shape", preds.shape)
        preds = top_k_scores.expand_as(preds) + preds

    pass


if __name__ == "__main__":
    fire.Fire(predict)
