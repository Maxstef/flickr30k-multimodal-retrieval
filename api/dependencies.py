from functools import lru_cache

import pandas as pd
import torch

from app.config import APP_DATA_DIR, MODEL_PATH
from src.serving.model_loader import (
    load_caption_index,
    load_image_index,
    load_mini_clip_model,
    load_vocab,
)


@lru_cache(maxsize=1)
def get_vocab() -> dict:
    """
    Load and cache the vocabulary for API inference.
    """
    return load_vocab(APP_DATA_DIR)


@lru_cache(maxsize=1)
def get_model():
    """
    Load and cache the Mini-CLIP model for API inference.
    """
    vocab = get_vocab()

    return load_mini_clip_model(
        vocab_size=len(vocab),
        model_path=MODEL_PATH,
    )


@lru_cache(maxsize=1)
def get_caption_index() -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Load and cache caption metadata and embeddings.
    """
    return load_caption_index(APP_DATA_DIR)


@lru_cache(maxsize=1)
def get_image_index() -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Load and cache image metadata and embeddings.
    """
    return load_image_index(APP_DATA_DIR)