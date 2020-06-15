LAMEM_ROOT = "/home/anelise/scratch/datasets/LaMem"
MEMENTO_ROOT = "/home/anelise/datasets/memento/release_v1"
MEMENTO_METADATA_PATH = "memento_metadata.json"

BATCH_SIZE = 20
NUM_WORKERS = 20
USE_GPU = True
NUM_EPOCHS = 100
LOGDIR = "logs"
CKPT_DIR = "ckpt"
PREDS_DIR = "preds"
RESIZE = 256
CROP_SIZE = 224
N_FRAMES_FOR_FRAMES_MODEL = 23

# TODO: store the config as YAML. Load the config and save it in a "struct"
# class.
# override the config with **kwargs passed to the training function. Save the
# config as YAML again in the ckpt and/or in a separate file
# https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
