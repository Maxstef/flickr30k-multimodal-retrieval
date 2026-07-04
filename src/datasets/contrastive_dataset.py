import random

import torch
from torch.utils.data import Dataset

from src.text.tokenization import encode_caption


class Flickr30kContrastiveDataset(Dataset):
    """
    PyTorch Dataset for CLIP-style contrastive image-text learning.

    Each item returns:
        - image tensor
        - encoded caption tensor

    Since each Flickr30k image has multiple captions, one caption is randomly
    selected for each image when the sample is retrieved.
    """

    def __init__(self, hf_dataset, vocab, transform=None, max_length=32):
        """
        Args:
            hf_dataset: Hugging Face Dataset split containing images and captions.
            vocab: Token-to-index vocabulary.
            transform: Image transform applied to PIL images.
            max_length: Maximum encoded caption length.
        """
        self.hf_dataset = hf_dataset
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        """Return the number of unique images."""
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        Retrieve one image and one randomly selected caption.

        Other captions in the same batch act as implicit negative examples
        during contrastive training.
        """
        sample = self.hf_dataset[idx]

        image = sample["image"].convert("RGB")
        caption = random.choice(sample["caption"])

        if self.transform is not None:
            image = self.transform(image)

        caption_ids = encode_caption(
            caption,
            self.vocab,
            max_length=self.max_length,
        )

        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        return image, caption_tensor


class Flickr30kContrastiveEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for CLIP-style contrastive learning using
    precomputed image embeddings.

    Each item returns:
        - image embedding tensor
        - encoded caption tensor

    Since each Flickr30k image has multiple valid captions, one caption is
    randomly selected whenever the sample is retrieved.
    """

    def __init__(self, hf_dataset, image_features, vocab, max_length=32, caption_strategy="random"):
        """
        Args:
            hf_dataset: Hugging Face Dataset split containing captions and filenames.
            image_features: Dictionary mapping filename to image embedding.
            vocab: Token-to-index vocabulary.
            max_length: Maximum encoded caption length.
        """
        self.hf_dataset = hf_dataset
        self.image_features = image_features
        self.vocab = vocab
        self.max_length = max_length
        self.caption_strategy = caption_strategy

    def __len__(self):
        """Return the number of unique images."""
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        filename = sample["filename"]
        captions = sample["caption"]

        if self.caption_strategy == "first":
            caption = captions[0]
        elif self.caption_strategy == "random":
            caption = random.choice(captions)
        else:
            raise ValueError(
                "caption_strategy must be either 'random' or 'first'"
            )

        image_embedding = self.image_features[filename]

        if not isinstance(image_embedding, torch.Tensor):
            image_embedding = torch.tensor(image_embedding, dtype=torch.float32)
        else:
            image_embedding = image_embedding.float()

        caption_ids = encode_caption(
            caption,
            self.vocab,
            max_length=self.max_length,
        )

        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        return image_embedding, caption_tensor