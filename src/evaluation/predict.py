import torch
from tqdm import tqdm


def predict_binary_classifier(
    model,
    dataloader,
    device,
    threshold=0.5,
):
    """
    Generate probabilities and binary predictions for a binary classifier.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader returning inputs and labels.
        device: CPU, CUDA, or MPS device.
        threshold: Probability threshold used to convert probabilities into labels.

    Returns:
        Dictionary with:
            - probabilities
            - predictions
            - labels
    """
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for inputs, captions, labels in tqdm(dataloader, desc="Predicting"):
            inputs = inputs.to(device)
            captions = captions.to(device)

            logits = model(inputs, captions)
            probs = torch.sigmoid(logits)

            preds = (probs >= threshold).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "probabilities": all_probs,
        "predictions": all_preds,
        "labels": all_labels,
    }