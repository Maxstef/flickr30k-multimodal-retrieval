import torch
from torch.utils.data import Dataset

from src.text.tokenization import encode_caption


class Flickr30kBinaryDataset(Dataset):
    """
    PyTorch Dataset for binary image-text matching using Flickr30k.

    Each item returns:
        - image tensor
        - encoded caption tensor
        - binary label tensor
    """

    def __init__(self, pairs, hf_dataset, vocab, transform=None, max_length=32):
        """
        Args:
            pairs: DataFrame with filename, caption, and label columns.
            hf_dataset: Hugging Face Dataset split containing the original images.
            vocab: Token-to-index vocabulary.
            transform: Image transform applied to PIL images.
            max_length: Maximum encoded caption length.
        """
        self.pairs = pairs.reset_index(drop=True)
        self.hf_dataset = hf_dataset
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

        # Build a lookup table so images can be retrieved by filename
        # in constant time instead of searching the dataset for every sample.
        self.filename_to_index = {
            filename: idx
            for idx, filename in enumerate(hf_dataset["filename"])
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Retrieve a single training sample.

        This method is called automatically by the DataLoader whenever
        a new sample is required for building a mini-batch.
        """
        row = self.pairs.iloc[idx]

        filename = row["filename"]
        caption = row["caption"]
        label = row["label"]

        # Find the corresponding image in the original Hugging Face dataset.
        image_idx = self.filename_to_index[filename]
        image = self.hf_dataset[image_idx]["image"].convert("RGB")

        # Apply preprocessing required by the pretrained ResNet18 model.
        if self.transform is not None:
            image = self.transform(image)

        # Convert the caption into a fixed-length sequence of token IDs.
        caption_ids = encode_caption(
            caption,
            self.vocab,
            max_length=self.max_length,
        )

        # Convert data into PyTorch tensors.
        # Long tensors are required for embedding layers,
        # while labels are stored as float values for binary classification.
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, caption_tensor, label_tensor