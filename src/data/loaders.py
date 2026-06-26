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

    train_data = data.filter(
        lambda sample: sample["split"] == train_split
    )

    val_data = data.filter(
        lambda sample: sample["split"] == val_split
    )

    test_data = data.filter(
        lambda sample: sample["split"] == test_split
    )

    return train_data, val_data, test_data


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