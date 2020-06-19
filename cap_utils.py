import json
import os
from collections import Counter

import numpy as np
from tqdm import tqdm

import config as cfg


def prepare_caption_data(tokenized_captions_json_path,
                         word_embeddings=None,
                         return_backward=False,
                         caption_format='index_list',
                         input_format=None,
                         vocab_size=cfg.VOCAB_SIZE,
                         max_cap_len=cfg.MAX_CAP_LEN,
                         add_padding=True):
    '''
    Prepares tokenized captions from a json to be fed to a captioning model.

    Inputs
    ------
        tokenized_captions_json_path: string, path to tokenized captions json.
        word_embeddings: path to word embeddings or pre-loaded embedding dict
        caption_format: string that determines the format and type of the
         return variables. Can take the following values:
            'index_triangular_matrix'
            'embedding_triangular_matrix'
            'index_list'
            'embedding_list'
            'one_hot_list'

    Returns
    -------
        input_captions: dictionary mapping a video name to the caption input of
        the model. Multiple different arrays can be returned depending on the
        value of caption_format.

        target_captions: dictionary mapping a video name to the desired target
        of the captioning task for that video. Changes depending on the value
        of caption_format.
    '''

    if input_format is None:
        input_format = caption_format

    if type(word_embeddings) is str:
        with open(word_embeddings) as infile:
            word_embeddings = json.load(infile)

    tokenized_captions = json.load(open(tokenized_captions_json_path, 'r'))
    print(len(tokenized_captions))

    input_captions = {}
    target_captions = {}

    for vid_name, cap_dict in tqdm(tokenized_captions.items()):

        input_captions[vid_name] = []
        target_captions[vid_name] = []

        for i, cap in enumerate(cap_dict['indexed_captions']):
            tokenized_cap = cap_dict['tokenized_captions'][i]

            in_cap, target_cap = transform_caption(
                cap,
                tokenized_cap,
                input_format,
                caption_format,
                add_padding=add_padding,
                word_embeddings=word_embeddings,
                max_cap_len=max_cap_len,
                vocab_size=vocab_size)
            input_captions[vid_name].append(in_cap)
            target_captions[vid_name].append(target_cap)

    return input_captions, target_captions


def transform_caption(cap,
                      tokenized_cap,
                      input_format,
                      caption_format,
                      word_embeddings,
                      max_cap_len,
                      vocab_size,
                      add_padding=True):

    shift_left = lambda c: c[1:] + [0]
    remove_padding = lambda c: [elt for elt in c if elt != 0]

    # prepare input data
    input_cap = cap
    input_tokenized_cap = tokenized_cap

    if not add_padding:
        input_cap = remove_padding(cap)
        # remove the end token from
        input_cap = cap[:-1]
        input_tokenized_cap = tokenized_cap[:-1]

    # handle input captions

    if input_format == 'index_triangular_matrix':
        input_cap = prepare_as_triangular(input_cap)
    elif input_format == 'embedding_triangular_matrix':
        input_cap = prepare_as_triangular(input_cap, embedding=word_embeddings)
    elif input_format == 'embedding_list':
        input_cap = transform_into_embedding(input_tokenized_cap,
                                             word_embeddings,
                                             max_cap_len=max_cap_len,
                                             offset_by_one=False,
                                             add_padding=add_padding)
    elif input_format == 'index_list':
        pass
    elif input_format == 'one_hot_list':
        input_cap = _to_categorical(input_cap, num_classes=vocab_size)
    else:
        raise ValueError("Unknown caption format for %s: %s" %
                         ("input_format", input_format))

    # prepare output data
    output_cap = cap
    output_tokenized_cap = tokenized_cap

    if not add_padding:
        output_cap = remove_padding(cap)
        output_cap = output_cap[1:]
        output_tokenized_cap = tokenized_cap[1:]
    else:
        # shift the whole thing left
        output_cap = shift_left(output_cap)

    # handle target captions

    if caption_format == 'index_triangular_matrix':
        output_cap = output_cap
    elif caption_format == 'embedding_triangular_matrix':
        output_cap = transform_into_embedding(output_tokenized_cap,
                                              word_embeddings,
                                              max_cap_len=max_cap_len)
    elif caption_format == 'embedding_list':
        output_cap = transform_into_embedding(output_tokenized_cap,
                                              word_embeddings,
                                              max_cap_len=max_cap_len)
    elif caption_format == 'index_list':
        output_cap = output_cap
    elif caption_format == 'one_hot_list':
        output_cap = _to_categorical(output_cap, num_classes=vocab_size)
    else:
        raise ValueError("Unknown caption format for %s: %s" %
                         ("caption_format", caption_format))

    return input_cap, output_cap


def _to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


def transform_into_embedding(list_of_words,
                             embedding,
                             offset_by_one=True,
                             max_cap_len=cfg.MAX_CAP_LEN,
                             add_padding=True):

    emb_list = []

    emb_len = len(embedding[list_of_words[0]])

    c = 1 if offset_by_one else 0

    for l in list_of_words[c:]:
        vec = embedding[l]
        assert vec is not None
        emb_list.append(vec)

    if add_padding:
        for i in range(max_cap_len - len(list_of_words) + c):
            emb_list.append([0] * emb_len)

    return np.array(emb_list)


def prepare_as_triangular(cap, return_backward=False, embedding=None):
    cap_len = next((i for i, x in enumerate(cap) if x == 0), cfg.MAX_CAP_LEN)

    if embedding is not None:
        cap = transform_into_embedding(cap, embedding, offset_by_one=False)
    try:
        cap_tiled = np.tile(cap, (cap_len - 1, 1))
    except:
        print(len(cap))
        print(cap_len)
        print(cap)

    # Diagonalizing for forward direction
    cap_matrix_forw = np.tril(cap_tiled)

    if return_backward:
        cap_tiled_bw = np.tile(cap[:cap_len], (cap_len, 1))
        # Diagonalizing for backward direction
        cap_matrix_backw = np.triu(cap_tiled_bw).fliplr()[::-1]

        return cap_matrix_forw, cap_matrix_backw

    return cap_matrix_forw
