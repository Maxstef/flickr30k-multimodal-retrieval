import torch
from torch import nn

from src.config import PAD_IDX


class NeuralMultimodalEmbeddingClassifier(nn.Module):
    """
    Neural classifier for binary image-text matching using precomputed image embeddings.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        image_feature_dim=512,
        hidden_dim=256,
        dropout=0.3,
    ):
        super().__init__()

        self.text_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image_embeddings, captions):
        text_embeddings = self.text_embedding(captions)

        mask = captions != PAD_IDX
        mask = mask.unsqueeze(-1)

        text_embeddings = text_embeddings * mask

        text_lengths = mask.sum(dim=1).clamp(min=1)
        text_features = text_embeddings.sum(dim=1) / text_lengths

        combined_features = torch.cat(
            [image_embeddings, text_features],
            dim=1,
        )

        logits = self.classifier(combined_features)

        return logits.squeeze(1)
