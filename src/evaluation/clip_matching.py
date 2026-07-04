import torch
from tqdm import tqdm


def compute_clip_pair_similarities(
    model,
    dataloader,
    device,
):
    """
    Compute cosine similarity scores for image-caption pairs using Mini-CLIP.

    Args:
        model: Trained Mini-CLIP model.
        dataloader: DataLoader returning image embeddings, captions, and labels.
        device: CPU, CUDA, or MPS device.

    Returns:
        Dictionary with:
            - similarities
            - labels
    """
    model.eval()

    all_similarities = []
    all_labels = []

    with torch.inference_mode():
        for image_embeddings, captions, labels in tqdm(
            dataloader,
            desc="Computing similarities",
        ):
            image_embeddings = image_embeddings.to(device)
            captions = captions.to(device)

            image_proj, text_proj = model(image_embeddings, captions)

            similarities = (image_proj * text_proj).sum(dim=1)

            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "similarities": all_similarities,
        "labels": all_labels,
    }