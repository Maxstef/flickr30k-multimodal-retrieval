"""Cached resource loaders for the Streamlit application."""

import pandas as pd
import streamlit as st
import torch

from app.config import APP_DATA_DIR, MODEL_PATH
from src.models.mini_clip import MiniCLIP
from src.serving.model_loader import (
    load_caption_index as load_caption_index_resources,
    load_image_index as load_image_index_resources,
    load_mini_clip_model as load_mini_clip_model_resource,
    load_vocab as load_vocab_resource,
)


@st.cache_resource
def load_mini_clip_model(vocab_size: int) -> MiniCLIP:
    """
    Load and cache the trained Mini-CLIP model.

    Args:
        vocab_size:
            Number of tokens in the vocabulary.

    Returns:
        MiniCLIP:
            Model configured for inference.
    """
    return load_mini_clip_model_resource(
        vocab_size=vocab_size,
        model_path=MODEL_PATH,
    )


@st.cache_data
def load_caption_index() -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Load and cache caption metadata and embeddings.

    Returns:
        tuple[pd.DataFrame, torch.Tensor]:
            Caption metadata and normalized caption embeddings.
    """
    return load_caption_index_resources(APP_DATA_DIR)


@st.cache_data
def load_vocab() -> dict:
    """
    Load and cache the Mini-CLIP vocabulary.

    Returns:
        dict:
            Vocabulary used by the text encoder.
    """
    return load_vocab_resource(APP_DATA_DIR)


@st.cache_data
def load_image_index() -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Load and cache image metadata and embeddings.

    Returns:
        tuple[pd.DataFrame, torch.Tensor]:
            Image metadata and normalized image embeddings.
    """
    return load_image_index_resources(APP_DATA_DIR)