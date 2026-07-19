"""Retrieval utilities for Mini-CLIP embeddings."""

import torch


@torch.inference_mode()
def retrieve_top_k(
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    top_k: int = 3,
) -> list[dict[str, float]]:
    """
    Retrieve the top-k most similar embeddings using cosine similarity.

    Assumes both query and candidate embeddings are L2-normalized.

    Args:
        query_embedding:
            Query embedding of shape [1, embedding_dim].
        candidate_embeddings:
            Candidate embeddings of shape [N, embedding_dim].
        top_k:
            Number of nearest neighbours to return.

    Returns:
        List of dictionaries containing the retrieved index and similarity score.
    """
    similarities = query_embedding @ candidate_embeddings.T

    scores, indices = similarities.squeeze(0).topk(top_k)

    return [
        {
            "index": index.item(),
            "score": score.item(),
        }
        for score, index in zip(scores, indices)
    ]
