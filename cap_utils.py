import json

import numpy as np
import torch
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


def index_vocab():
    with open(cfg.MEMENTO_VOCAB) as infile:
        vocab = json.load(infile)
    word2idx = {word: i + 1 for i, word in enumerate(vocab)}
    idx2word = {i + 1: word for i, word in enumerate(vocab)}
    idx2word[0] = "<end>"

    return word2idx, idx2word


def one_hot_to_token(one_hot, idx2word):
    idx = np.argmax(one_hot)
    token = idx2word[idx]

    return token


def index_to_token(index, idx2word):
    return idx2word[index]


def get_vocab_weights(eps=0.0001):
    return eps + np.load(cfg.MEMENTO_VOCAB_WEIGHTS)


def get_vocab_embedding():
    with open(cfg.MEMENTO_CAPTIONS_EMBEDDING) as infile:
        vocab_embedding = json.load(infile)

    return vocab_embedding


def predict_captions_simple(model, x, device, vocab_embedding, idx2word):

    features, feature_map = model.module.encode(x)  # (batch(1), 1024, 5, 1, 1)
    batch_size = features.shape[0]

    start_token_embedded = vocab_embedding['<start>']
    inp = torch.Tensor([start_token_embedded] * batch_size)
    inp = inp.to(device)

    # initialize the lstm
    h, c = model.module.init_hidden_state(features)

    words = []
    att_alphas = []

    for step in range(cfg.MAX_CAP_LEN):
        h, c, out, att_alphas = model.module.caption_decode_step(
            h, c, inp, feature_map, return_alphas=True)
        out_numpy = out.to("cpu").numpy()
        token = one_hot_to_token(out_numpy, idx2word)
        inp = torch.Tensor([vocab_embedding[token]]).to(device)
        words.append(token)

    return words


def predict_captions_beam(model,
                          x: torch.Tensor,
                          device,
                          vocab_embedding,
                          idx2word,
                          beam_size: int = 3) -> None:
    """IN PROGRESS"""
    k = beam_size
    vocab_size = cfg.VOCAB_SIZE

    # seed your list of captions;
    # holds the top k sequences
    seq_tokens = np.array([['<start>'] for _ in range(k)])

    prev_tokens = seq_tokens[:, 0]

    # Setup #################################################

    # holds the scores for top k beam search contenders
    # At every it, you add the log prob of the next word to
    # calculate the score of the whole seq.
    top_k_scores = torch.zeros(k, 1).to(device)  # (k,1)

    # Running the search ####################################

    # encode the image
    features = model.module.encode(x)  # (batch(1), 1024, 5, 1, 1)

    # duplicate the features k times
    newshape = [k] + list(features.shape)[1:]
    features = features.expand(newshape)  # (k, 1024, 5, 1, 1)

    # initialize the lstm
    h, c = model.module.init_hidden_state(features)

    for step in range(cfg.MAX_CAP_LEN):
        # tokens to embedding
        prev_words_embedded = torch.Tensor(
            [vocab_embedding[token] for token in prev_tokens])
        prev_words_embedded = prev_words_embedded.to(device)

        h, c, out = model.module.caption_decode_step(h, c, prev_words_embedded)
        scores = torch.nn.functional.log_softmax(out, dim=1)
        # add the current probs output by the lstm to the old probs
        scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)

        if step == 0:
            # All k scores are the same; just take the top k start words
            # from the first dimension (scores[0])
            # shape (k,)
            top_k_scores, top_k_indices = scores[0].topk(k,
                                                         dim=0,
                                                         largest=True,
                                                         sorted=True)
        else:
            # Flatten each word for each beam into a long list and get
            # top k
            # shape (k,)
            top_k_scores, top_k_indices = scores.view(-1).topk(k,
                                                               dim=0,
                                                               largest=True,
                                                               sorted=True)
        top_k_scores.unsqueeze_(1)

        # out of the k growing sequences, which one does this word belong to?
        prev_word_inds = (top_k_indices / vocab_size)
        prev_seq_tokens = seq_tokens[prev_word_inds.to("cpu").numpy()]
        # what is the next word we are going to add on?
        next_word_inds = (top_k_indices % vocab_size).to("cpu").numpy()
        next_tokens = np.array(
            [index_to_token(index, idx2word) for index in next_word_inds])

        seq_tokens = np.hstack(
            (prev_seq_tokens, np.expand_dims(next_tokens, axis=1)))

        # rearrange the lstm inner state and set up the next round
        h = h[prev_word_inds]
        c = c[prev_word_inds]
        prev_tokens = next_tokens

    # TODO: choose the best sequence

    return seq_tokens
