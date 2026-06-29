from collections import defaultdict
from datasets import load_dataset


def load_flickr30k_splits(
    dataset_name,
    train_split="train",
    val_split="val",
    test_split="test",
):
    """
    Load the Flickr30k dataset and return the original train,
    validation, and test splits.

    The Hugging Face dataset stores all samples under the "test" split,
    while the original split is stored in the "split" column.

    Args:
        dataset_name: Hugging Face dataset name.
        train_split: Name of the training split.
        val_split: Name of the validation split.
        test_split: Name of the test split.

    Returns:
        A tuple containing:
            - train_data
            - val_data
            - test_data
    """
    dataset = load_dataset(dataset_name)
    data = dataset["test"]

    split_to_indices = defaultdict(list)

    # Scan the split column once and group sample indices by split.
    # Using `select()` is considerably faster than filtering the dataset three separate times.
    for idx, split in enumerate(data["split"]):
        split_to_indices[split].append(idx)

    return (
        data.select(split_to_indices[train_split]),
        data.select(split_to_indices[val_split]),
        data.select(split_to_indices[test_split]),
    )


def load_flickr30k(dataset_name):
    """
    Load the Flickr30k dataset.

    The Hugging Face version stores all samples in the "test" split,
    while the original dataset split is provided in the "split" column.

    Args:
        dataset_name: Hugging Face dataset name.

    Returns:
        Hugging Face Dataset containing all Flickr30k samples.
    """
    dataset = load_dataset(dataset_name)

    return dataset["test"]