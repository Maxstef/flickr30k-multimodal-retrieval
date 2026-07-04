import torch
from torch import nn
import torch.nn.functional as F

from src.config import PAD_IDX


class MiniCLIP(nn.Module):
    """
    Lightweight CLIP-style model for image-text representation learning.

    The model receives precomputed image embeddings and encoded captions,
    then projects both modalities into a shared embedding space.
    """

    def __init__(
        self,
        vocab_size,
        text_embedding_dim=128,
        image_feature_dim=512,
        projection_dim=256,
    ):
        super().__init__()

        self.image_projection = nn.Linear(
            image_feature_dim,
            projection_dim,
        )

        self.text_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=text_embedding_dim,
            padding_idx=PAD_IDX,
        )

        self.text_projection = nn.Linear(
            text_embedding_dim,
            projection_dim,
        )

    def forward(self, image_features, captions):
        """
        Encode image features and captions into a shared embedding space.

        Args:
            image_features: Batch of precomputed image embeddings.
            captions: Batch of encoded caption tensors.

        Returns:
            Normalized image and text embeddings.
        """
        image_embeddings = self.image_projection(image_features)

        text_embeddings = self.text_embedding(captions)

        # Ignore padding tokens when averaging word embeddings.
        mask = captions != PAD_IDX
        mask = mask.unsqueeze(-1)

        text_embeddings = text_embeddings * mask

        text_lengths = mask.sum(dim=1).clamp(min=1)
        text_features = text_embeddings.sum(dim=1) / text_lengths

        text_embeddings = self.text_projection(text_features)

        # Normalize embeddings so dot product becomes cosine similarity.
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

        return image_embeddings, text_embeddings