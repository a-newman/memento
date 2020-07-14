import json
import os
import sys

import numpy as np

try:
    sys.path.append("..")
    import config as cfg
except:
    raise RuntimeError()


def get_weights():
    with open(os.path.join(cfg.MEMENTO_ROOT,
                           "memento_train_data.json")) as infile:
        data = json.load(infile)

    mem_scores = [elt['mem_score'] for elt in data]

    hist, edges = np.histogram(mem_scores, bins=21, range=(0, 1.05))
    print(hist)
    print(edges)
    eps = .1
    print("len hist", len(hist))
    # weights = 1 / len(hist) * (1 / (hist + eps))
    # weights = 1 - hist / (sum(hist))
    weights = hist / sum(hist)
    print("weights", weights)
    print("sum", np.sum(weights * hist))

    np.save("memento_weights.npy", weights)


if __name__ == "__main__":
    get_weights()
