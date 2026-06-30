import torch
from pathlib import Path

# -----------------------------------------------------------------------------
# Project directories
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

# Hugging Face dataset identifier
DATASET_NAME = "nlphuji/flickr30k"

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# -----------------------------------------------------------------------------
# Text
# -----------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1
VOCAB_MIN_FREQUENCY = 1
MAX_CAPTION_LENGTH = 32
EMBEDDING_DIM = 128


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

IMAGE_FEATURE_DIM = 512
HIDDEN_DIM = 256
DROPOUT = 0.3
