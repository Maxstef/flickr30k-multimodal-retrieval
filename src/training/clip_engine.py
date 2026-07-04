import torch
from tqdm import tqdm


def train_clip_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Train a CLIP-style model for one epoch.
    """
    model.train()

    total_loss = 0

    for image_embeddings, captions in tqdm(dataloader, desc="Training"):
        image_embeddings = image_embeddings.to(device)
        captions = captions.to(device)

        image_proj, text_proj = model(image_embeddings, captions)

        loss = loss_fn(image_proj, text_proj)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_clip(model, dataloader, loss_fn, device):
    """
    Evaluate a CLIP-style model using contrastive loss.
    """
    model.eval()

    total_loss = 0

    with torch.inference_mode():
        for image_embeddings, captions in tqdm(dataloader, desc="Evaluating"):
            image_embeddings = image_embeddings.to(device)
            captions = captions.to(device)

            image_proj, text_proj = model(image_embeddings, captions)

            loss = loss_fn(image_proj, text_proj)

            total_loss += loss.item()

    return total_loss / len(dataloader)