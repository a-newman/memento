import json
import os
from typing import Callable, Mapping, Optional

import fire
import nltk
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import cap_utils
import config as cfg
import utils
from data_loader import (VIDEO_TEST_TRANSFORMS, VIDEO_TRAIN_TRANSFORMS,
                         get_dataset)
from model_utils import MemModelFields, ModelOutput
from models import get_model


def rc(labels: ModelOutput[MemModelFields],
       preds: ModelOutput[MemModelFields],
       _,
       verbose=True):
    # (b, t=max cap len, d=vocab size)
    mem_scores_true = labels['score']
    mem_scores_pred = preds['score']
    val = spearmanr(mem_scores_true, mem_scores_pred)

    return val.correlation


def bleu_score(labels: ModelOutput[MemModelFields],
               preds: ModelOutput[MemModelFields], _):
    """Calculates the BLEU score using only the caption that was passed in
    (for simplicity), not all 5.

    To be used while training to evaluate model performance.
    """

    if not ('out_captions' in labels and 'out_captions' in preds):
        return 0

    # (b=batch, t=cap len, d=vocab size)
    ref_cap_one_hot = np.argmax(labels['out_captions'], axis=2)  # (b,t)
    pred_cap_one_hot = np.argmax(preds['out_captions'], axis=2)

    # we're going to pretend the numbers themselves are the tokens
    # and not bother translating
    # TODO: trim caps after <end>
    scores = []

    for b in range(len(ref_cap_one_hot)):
        ref = ref_cap_one_hot[b, :]
        pred = pred_cap_one_hot[b, :]
        scores.append(nltk.translate.bleu_score.sentence_bleu(ref, pred))

    return np.mean(scores)

    # sentence_bleu(reference_captions, candidate)


def predict(
    ckpt_path,
    should_predict_captions: bool = False,
    # metrics: Mapping[str, Callable] = {
    #     'rc': rc,
    # },
    num_workers: int = 20,
    use_gpu: bool = True,
    model_name: str = "frames",
    dset_name: str = "memento_frames",
    batch_size: int = 1,
    preds_savepath: Optional[str] = None,
    use_val: bool = False,
    debug_n: Optional[int] = None,
    n_mem_repetitions=5):

    print("ckpt path: {}".format(ckpt_path))

    if preds_savepath is None:
        fname = "_" + ("captions"
                       if should_predict_captions else "mems") + ".json"
        preds_savepath = os.path.splitext(
            ckpt_path.replace(cfg.CKPT_DIR, cfg.PREDS_DIR))[0] + fname
        utils.makedirs([os.path.dirname(preds_savepath)])
    print("preds savepath: {}".format(preds_savepath))

    device = utils.set_device()
    print('DEVICE', device)

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
    use_augmentations = (not should_predict_captions) and (n_mem_repetitions >
                                                           1)
    print("Use augmentations?", use_augmentations)
    test_transforms = VIDEO_TRAIN_TRANSFORMS if use_augmentations else VIDEO_TEST_TRANSFORMS
    train, val, test = get_dataset(dset_name, test_transforms=test_transforms)
    ds = val if use_val else test

    if ds is None:
        raise ValueError("No {} set available for this dataset.".format(
            "val" if use_val else "test"))
    ordered_fnames = ds.get_fnames()

    if debug_n is not None:
        ds = Subset(ds, range(debug_n))

    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)

    # either do mem scores or captions

    if should_predict_captions:
        # load the vocab embedding
        with open(cfg.MEMENTO_CAPTIONS_EMBEDDING) as infile:
            vocab_embedding = json.load(infile)

        # load the vocab itself
        word2idx, idx2word = cap_utils.index_vocab()

        calc_captions(model, dl, ds, batch_size, device, vocab_embedding,
                      idx2word, preds_savepath, ordered_fnames)

    else:
        _calc_mem_scores(model,
                         ckpt_path,
                         dl,
                         ds,
                         batch_size,
                         device,
                         preds_savepath,
                         n_times=n_mem_repetitions)


def _calc_mem_scores(model, ckpt_path, dl, ds, batch_size, device,
                     preds_savepath, n_times):

    alphas = []
    mems = []
    gt_mems = None
    gt_alphas = None

    for i in range(n_times):
        print("Generating mem scores, round {}".format(i))

        preds: Optional[ModelOutput] = None
        labels: Optional[ModelOutput] = None
        with torch.no_grad():
            for i, (x, y_) in tqdm(enumerate(dl), total=len(ds) / batch_size):

                y: ModelOutput[MemModelFields] = ModelOutput(y_)
                y_list = y.to_numpy()
                labels = y_list if labels is None else labels.merge(y_list)

                x = x.to(device)
                y = y.to_device(device)

                out = ModelOutput(model(x, y.get_data()))

                out_list = out.to_device('cpu').to_numpy()
                preds = out_list if preds is None else preds.merge(out_list)

        mems.append(preds['score'])
        alphas.append(preds['alpha'])

        print("correlation", spearmanr(preds['score'], labels['score']))

        if gt_mems is None:
            gt_mems = labels['score']
            gt_alphas = labels['alpha']

    # merge mem scores
    mems_avg = np.array(mems).mean(axis=0)
    alphas_avg = np.array(alphas).mean(axis=0)

    rc_value = spearmanr(mems_avg, gt_mems)
    print("rc", rc_value)

    metrics = {'rc': rc_value.correlation}

    data = {
        'ckpt': ckpt_path,
        'mems': mems_avg.tolist(),
        'alphas': alphas_avg.tolist(),
        'gt_mems': gt_mems.tolist(),
        'gt_alphas': gt_alphas.tolist(),
        'metrics': metrics
    }

    with open(preds_savepath, "w") as outfile:
        print("Saving results")
        json.dump(data, outfile)


def calc_captions(model, dl, ds, batch_size, device, vocab_embedding, idx2word,
                  preds_savepath, fnames):
    assert batch_size == 1
    captions = {}
    with torch.no_grad():
        for i, (x, y_) in tqdm(enumerate(dl), total=len(ds) / batch_size):

            y: ModelOutput[MemModelFields] = ModelOutput(y_)

            x = x.to(device)
            y = y.to_device(device)

            words = cap_utils.predict_captions_simple(model, x, device,
                                                      vocab_embedding,
                                                      idx2word)
            # words_beam = cap_utils.predict_captions_beam(model, x, device,
            #                                    vocab_embedding, idx2word)

            fname = fnames[i]
            captions[fname] = words
            print("simple", words)
            # print("beam", words_beam)

    with open(preds_savepath, "w") as outfile:
        print("saving results")
        json.dump(captions, outfile)


if __name__ == "__main__":
    fire.Fire(predict)
