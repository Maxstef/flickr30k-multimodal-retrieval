import random
import matplotlib.pyplot as plt


def show_sample(dataset, idx, split="test", figsize=(8, 8)):
    """
    Display an image together with all of its associated captions.

    Args:
        dataset: Flickr30k dataset loaded with Hugging Face Datasets.
        idx: Index of the sample to display.
        split: Dataset split to sample from ("train", "val", or "test").
        figsize: Size of the matplotlib figure as (width, height).

    Returns:
        None.
    """
    sample = dataset[split][idx]

    plt.figure(figsize=figsize)
    plt.imshow(sample["image"])
    plt.axis("off")
    plt.show()

    print(f"Index: {idx}")
    print(f"Image ID: {sample['img_id']}")
    print(f"Filename: {sample['filename']}\n")

    for i, caption in enumerate(sample["caption"], start=1):
        print(f"{i}. {caption}")


def get_random_indices(dataset, split="test", n=3, seed=42):
    """
    Generate reproducible random sample indices.

    Args:
        dataset: Flickr30k dataset loaded with Hugging Face Datasets.
        split: Dataset split to sample from ("train", "val", or "test").
        n: Number of random indices to generate.
        seed: Random seed for reproducibility.

    Returns:
        A list of randomly selected sample indices.
    """
    random.seed(seed)
    return random.sample(range(len(dataset[split])), n)