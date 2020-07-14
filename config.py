import os

LAMEM_ROOT = "/home/anelise/scratch/datasets/LaMem"
MEMENTO_ROOT = "/home/anelise/datasets/memento/release_v1"
MEMENTO_METADATA_PATH = "memento_metadata.json"
MEMENTO_CAPTIONS_PATH = "/home/anelise/datasets/memento/captions/cutoff_freq_10/"
MEMENTO_CAPTIONS_DATA = "memento_captions.json"
MEMENTO_CAPTIONS_EMBEDDING = os.path.join(MEMENTO_CAPTIONS_PATH,
                                          "vocab_embedding.json")
MEMENTO_VOCAB = os.path.join(MEMENTO_CAPTIONS_PATH, "vocab.json")
MEMENTO_VOCAB_WEIGHTS = os.path.join(MEMENTO_CAPTIONS_PATH, "vocab_freq.npy")

# Only if you want pretrained weights for the ResNet3D-34
RESNET_3D_34_PRETRAINED_CKPT = "r3d34_K_200ep.pth"

BATCH_SIZE = 32
NUM_WORKERS = 20
USE_GPU = True
NUM_EPOCHS = 100
DATA_SAVEDIR = "data"
LOGDIR = "logs"
CKPT_DIR = "ckpt"
PREDS_DIR = "preds"
RESIZE = 256
CROP_SIZE = 224
N_FRAMES_FOR_FRAMES_MODEL = 23
VOCAB_SIZE = 2547
MAX_CAP_LEN = 50

# TODO: store the config as YAML. Load the config and save it in a "struct"
# class.
# override the config with **kwargs passed to the training function. Save the
# config as YAML again in the ckpt and/or in a separate file
# https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
