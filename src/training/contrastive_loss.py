import torch
from torch import nn
import torch.nn.functional as F


class CLIPContrastiveLoss(nn.Module):
    """
    Symmetric contrastive loss for CLIP-style image-text learning.
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Scaling factor for similarity scores.
                Lower values make the contrastive task sharper.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        """
        Compute symmetric image-text contrastive loss.

        Args:
            image_embeddings: Normalized image embeddings of shape [batch_size, dim].
            text_embeddings: Normalized text embeddings of shape [batch_size, dim].

        Returns:
            Scalar contrastive loss.
        """
        logits = image_embeddings @ text_embeddings.T
        logits = logits / self.temperature

        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        image_to_text_loss = F.cross_entropy(logits, labels)
        text_to_image_loss = F.cross_entropy(logits.T, labels)

        loss = (image_to_text_loss + text_to_image_loss) / 2

        return loss