import torch


def search_top_k(query_embedding, candidate_embeddings, top_k=3):
    """
    Search top-k nearest embeddings using cosine similarity.

    Assumes embeddings are already normalized.
    """
    similarities = query_embedding @ candidate_embeddings.T

    scores, indices = similarities.squeeze(0).topk(top_k)

    return [
        {
            "index": idx.item(),
            "score": score.item(),
        }
        for score, idx in zip(scores, indices)
    ]