import torch
import pandas as pd


def top_k_accuracy(similarity_matrix, k=1):
    """
    Compute Top-K retrieval accuracy.

    The correct match for item i is assumed to be item i.
    """
    top_k_indices = similarity_matrix.topk(k, dim=1).indices
    targets = torch.arange(similarity_matrix.shape[0]).unsqueeze(1)

    correct = (top_k_indices == targets).any(dim=1)

    return correct.float().mean().item()


def compute_retrieval_metrics(similarity_matrix):
    """
    Compute image-to-text and text-to-image retrieval metrics.
    """
    return pd.DataFrame(
        [
            {
                "direction": "Image-to-text",
                "top_1": top_k_accuracy(similarity_matrix, k=1),
                "top_5": top_k_accuracy(similarity_matrix, k=5),
                "top_10": top_k_accuracy(similarity_matrix, k=10),
            },
            {
                "direction": "Text-to-image",
                "top_1": top_k_accuracy(similarity_matrix.T, k=1),
                "top_5": top_k_accuracy(similarity_matrix.T, k=5),
                "top_10": top_k_accuracy(similarity_matrix.T, k=10),
            },
        ]
    )


def image_to_text_top_k_accuracy_multi_caption(
    similarity_matrix,
    image_indices,
    k=1,
):
    """
    Compute image-to-text Top-K retrieval accuracy when each image
    can have multiple valid captions.

    Args:
        similarity_matrix: Tensor of shape [num_images, num_captions].
        image_indices: Tensor/list mapping each caption index to its image index.
        k: Number of retrieved captions considered.

    Returns:
        Top-K accuracy.
    """
    image_indices = torch.tensor(image_indices)

    top_k_caption_indices = similarity_matrix.topk(k, dim=1).indices

    correct = []

    for image_idx, retrieved_caption_indices in enumerate(top_k_caption_indices):
        retrieved_image_indices = image_indices[retrieved_caption_indices]
        correct.append((retrieved_image_indices == image_idx).any())

    return torch.tensor(correct).float().mean().item()
