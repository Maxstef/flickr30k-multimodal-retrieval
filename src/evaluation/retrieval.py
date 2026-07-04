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
