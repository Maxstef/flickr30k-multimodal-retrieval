import torch
from torch.utils.data import Dataset

from src.text.tokenization import encode_caption


class Flickr30kEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for binary image-text matching using precomputed image embeddings.

    Each item returns:
        - image embedding tensor
        - encoded caption tensor
        - binary label tensor
    """

    def __init__(self, pairs, image_features, vocab, max_length=32):
        """
        Args:
            pairs: DataFrame with filename, caption, and label columns.
            image_features: Dictionary mapping filenames to precomputed image embeddings.
            vocab: Token-to-index vocabulary.
            max_length: Maximum encoded caption length.
        """
        self.pairs = pairs.reset_index(drop=True)
        self.image_features = image_features
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        """Return the total number of image-caption pairs."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Retrieve a single training sample.

        The image representation is loaded from precomputed ResNet18 embeddings
        instead of passing the image through the CNN during training.
        """
        row = self.pairs.iloc[idx]

        filename = row["filename"]
        caption = row["caption"]
        label = row["label"]

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
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_embedding, caption_tensor, label_tensor
