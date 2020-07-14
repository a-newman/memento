import json
import os
import pickle
import sys

import fire
import tqdm

try:
    sys.path.append('..')
    from cap_utils import prepare_caption_data
    import config as cfg
except:
    raise ImportError


def reverse_reformat(vid):
    idx = vid.index("/")
    vid_prefix = vid[:idx]
    vid_prefix = vid_prefix.replace("+", "-")
    vid = vid_prefix + "_" + vid[idx + 1:]
    vid = os.path.splitext(vid)[0]

    return vid


def generate_caption_data(savepath):
    all_cap_data = {}

    for split in ["train", "val", "test"]:
        caps_path = os.path.join(
            cfg.MEMENTO_CAPTIONS_PATH,
            "memento_{}_tokenized_captions.json".format(split))

        with open(caps_path) as infile:
            caps_data = json.load(infile)

        caps_data = {reverse_reformat(k): v for k, v in caps_data.items()}

        all_cap_data.update(caps_data)

        # embedding_path = os.path.join(cfg.memento_captions_path,
        #                               "vocab_embedding.json")

        # print("getting caption data")
        # input_captions, target_captions = prepare_caption_data(
        #     tokenized_captions_json_path=caps_path,
        #     word_embeddings=embedding_path,
        #     return_backward=false,
        #     caption_format='one_hot_list',
        #     input_format='embedding_list',
        #     add_padding=True)

        # for k in input_captions.keys():
        #     all_cap_data[k] = (input_captions[k], target_captions[k])

    # print(len(all_cap_data))

    with open(savepath, 'w') as outfile:
        json.dump(all_cap_data, outfile)


if __name__ == "__main__":
    fire.Fire(generate_caption_data)
