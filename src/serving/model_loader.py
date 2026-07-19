"""Framework-independent loaders for Mini-CLIP inference resources."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.config import (
    DEVICE,
    EMBEDDING_DIM,
    IMAGE_FEATURE_DIM,
    PROJECTION_DIM,
)
from src.models.mini_clip import MiniCLIP


def load_mini_clip_model(
    vocab_size: int,
    model_path: Path,
) -> MiniCLIP:
    """
    Load the trained Mini-CLIP model for inference.

    Args:
        vocab_size:
            Number of tokens in the vocabulary.
        model_path:
            Path to the trained Mini-CLIP state dictionary.

    Returns:
        MiniCLIP:
            Model configured for inference.
    """
    model = MiniCLIP(
        vocab_size=vocab_size,
        text_embedding_dim=EMBEDDING_DIM,
        image_feature_dim=IMAGE_FEATURE_DIM,
        projection_dim=PROJECTION_DIM,
    ).to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_caption_index(
    app_data_dir: Path,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Load caption metadata and precomputed caption embeddings.

    Args:
        app_data_dir:
            Directory containing the caption index files.

    Returns:
        tuple[pd.DataFrame, torch.Tensor]:
            Caption metadata and normalized caption embeddings.
    """
    captions_path = app_data_dir / "captions.csv"
    embeddings_path = app_data_dir / "caption_embeddings.pt"

    captions_df = pd.read_csv(captions_path)
    caption_embeddings = torch.load(
        embeddings_path,
        map_location="cpu",
    )

    return captions_df, caption_embeddings


def load_vocab(app_data_dir: Path) -> dict[str, Any]:
    """
    Load the vocabulary used by the Mini-CLIP text encoder.

    Args:
        app_data_dir:
            Directory containing the vocabulary file.

    Returns:
        dict[str, Any]:
            Vocabulary configuration loaded from JSON.
    """
    vocab_path = app_data_dir / "vocab.json"

    with vocab_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_image_index(
    app_data_dir: Path,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Load image metadata and precomputed image embeddings.

    Args:
        app_data_dir:
            Directory containing the image index files.

    Returns:
        tuple[pd.DataFrame, torch.Tensor]:
            Image metadata and normalized image embeddings.
    """
    images_path = app_data_dir / "images.csv"
    embeddings_path = app_data_dir / "image_embeddings.pt"

    images_df = pd.read_csv(images_path)
    image_embeddings = torch.load(
        embeddings_path,
        map_location="cpu",
    )

    return images_df, image_embeddings