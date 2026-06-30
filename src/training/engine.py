import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: CPU, CUDA, or MPS device.

    Returns:
        Average training loss for the epoch.
    """
    model.train()

    total_loss = 0

    for images, captions, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)
        labels = labels.to(device)

        logits = model(images, captions)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate the model on a validation or test set.

    Args:
        model: PyTorch model.
        dataloader: Validation or test DataLoader.
        loss_fn: Loss function.
        device: CPU, CUDA, or MPS device.

    Returns:
        A dictionary containing loss and classification metrics.
    """
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, captions, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            captions = captions.to(device)
            labels = labels.to(device)

            logits = model(images, captions)
            loss = loss_fn(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
