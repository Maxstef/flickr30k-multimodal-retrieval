"""Streamlit-facing wrappers around shared inference utilities."""

from PIL import Image
import torch

from src.models.mini_clip import MiniCLIP
from src.serving.inference import (
    encode_image,
    encode_text,
)


def encode_uploaded_image(
    image: Image.Image,
    model: MiniCLIP,
) -> torch.Tensor:
    """
    Encode an uploaded image into the Mini-CLIP embedding space.

    This wrapper preserves the existing Streamlit-facing function name.
    """
    return encode_image(
        image=image,
        model=model,
    )


def encode_text_query(
    text: str,
    model: MiniCLIP,
    vocab: dict[str, int],
) -> torch.Tensor:
    """
    Encode a text query into the Mini-CLIP embedding space.

    This wrapper preserves the existing Streamlit-facing function name.
    """
    return encode_text(
        text=text,
        model=model,
        vocab=vocab,
    )