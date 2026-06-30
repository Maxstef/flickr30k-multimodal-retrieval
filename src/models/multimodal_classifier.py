import torch
from torch import nn
from torchvision import models

from src.config import PAD_IDX


class NeuralMultimodalClassifier(nn.Module):
    """
    Neural network for binary image-text matching.

    The model consists of:
    - a pretrained ResNet18 image encoder,
    - a trainable text embedding layer,
    - a feed-forward classifier that predicts whether
      an image and caption match.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        image_feature_dim=512,
        hidden_dim=256,
        dropout=0.3,
        freeze_image_encoder=True,
    ):
        super().__init__()

        # ---------------------------------------------------------------------
        # Image encoder
        # ---------------------------------------------------------------------

        # Load a pretrained ResNet18 model.
        weights = models.ResNet18_Weights.DEFAULT
        self.image_encoder = models.resnet18(weights=weights)

        # Remove the final classification layer so that the network
        # outputs a 512-dimensional feature vector instead of ImageNet classes.
        self.image_encoder.fc = nn.Identity()

        # Freeze the image encoder during the first experiment.
        # This allows us to use pretrained visual features without
        # updating the ResNet18 weights.
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # ---------------------------------------------------------------------
        # Text encoder
        # ---------------------------------------------------------------------

        # Learn a dense vector representation for every word in the vocabulary.
        self.text_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,                  # Index of the <pad> token; its embedding remains zero and is not updated.
        )

        # ---------------------------------------------------------------------
        # Multimodal classifier
        # ---------------------------------------------------------------------

        # Concatenate image and text features and predict whether
        # they represent a matching image-caption pair.
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, images, captions):
        """
        Forward pass.

        Args:
            images: Batch of image tensors.
            captions: Batch of encoded caption tensors.

        Returns:
            One prediction logit for each image-caption pair.
        """

        # ---------------------------------------------------------------------
        # Image branch
        # ---------------------------------------------------------------------

        # Extract a 512-dimensional feature vector for every image.
        image_features = self.image_encoder(images)

        # ---------------------------------------------------------------------
        # Text branch
        # ---------------------------------------------------------------------

        # Convert token IDs into trainable word embeddings.
        # captions -> torch.Size([32, 32]) => text_embeddings -> torch.Size([32, 32, 128])
        text_embeddings = self.text_embedding(captions)

        # Ignore padding tokens when averaging embeddings
        mask = captions != PAD_IDX
        mask = mask.unsqueeze(-1)

        text_embeddings = text_embeddings * mask

        # Compute the average embedding for each caption.
        # This produces a single fixed-size representation regardless of caption length.
        # text_embeddings -> torch.Size([32, 32, 128]) => text_features -> torch.Size([32, 128])
        text_lengths = mask.sum(dim=1).clamp(min=1)
        text_features = text_embeddings.sum(dim=1) / text_lengths

        # ---------------------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------------------

        # Combine image and text representations into one feature vector.
        combined_features = torch.cat(
            [image_features, text_features],
            dim=1,
        )

        # ---------------------------------------------------------------------
        # Classification
        # ---------------------------------------------------------------------

        # Predict whether the image and caption match.
        logits = self.classifier(combined_features)

        # Remove the extra dimension to obtain one logit per sample.
        return logits.squeeze(1)
