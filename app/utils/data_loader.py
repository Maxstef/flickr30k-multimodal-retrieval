import sys

import pandas as pd
import streamlit as st
import torch
import json

from app.config import PROJECT_ROOT, MODEL_PATH, APP_DATA_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DEVICE,
    EMBEDDING_DIM,
    IMAGE_FEATURE_DIM,
    PROJECTION_DIM,
)
from src.models.mini_clip import MiniCLIP


@st.cache_resource
def load_mini_clip_model(vocab_size):
    model = MiniCLIP(
        vocab_size=vocab_size,
        text_embedding_dim=EMBEDDING_DIM,
        image_feature_dim=IMAGE_FEATURE_DIM,
        projection_dim=PROJECTION_DIM,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )
    )

    model.eval()
    return model


@st.cache_data
def load_caption_index():
    captions_df = pd.read_csv(APP_DATA_DIR / "captions.csv")
    caption_embeddings = torch.load(
        APP_DATA_DIR / "caption_embeddings.pt",
        map_location="cpu",
    )

    return captions_df, caption_embeddings

@st.cache_data
def load_vocab():
    with open(APP_DATA_DIR / "vocab.json", "r") as f:
        return json.load(f)
