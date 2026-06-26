import random
import pandas as pd


def extract_columns(split_data):
    """
    Extract commonly used columns from a Flickr30k dataset split.

    Args:
        split_data: Hugging Face Dataset.

    Returns:
        Tuple containing image IDs, filenames, and captions.
    """
    return (
        split_data["img_id"],
        split_data["filename"],
        split_data["caption"],
    )


def create_positive_pairs(split_data):
    """
    Create positive image-caption pairs from a Flickr30k dataset split.

    Each image is paired with its original human-written captions.
    Every generated pair receives the label 1.

    Args:
        split_data: Hugging Face Dataset containing a single split.

    Returns:
        A pandas DataFrame containing positive image-caption pairs.
    """
    positive_pairs = []

    img_ids, filenames, captions = extract_columns(split_data)

    # avoid for sample in split_data as it retrieves row by row from Hugging Face datasets and it is not efficient 
    for img_id, filename, caption_list in zip(img_ids, filenames, captions):
        for caption in caption_list:
            positive_pairs.append({
                "img_id": img_id,
                "filename": filename,
                "caption": caption,
                "label": 1,
            })

    return pd.DataFrame(positive_pairs)


def create_negative_pairs(split_data, seed=42):
    """
    Create negative image-caption pairs from a Flickr30k dataset split.

    Each image is paired with captions randomly selected from different images.
    Every generated pair receives the label 0.

    Args:
        split_data: Hugging Face Dataset containing a single split.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame containing negative image-caption pairs.
    """
    rng = random.Random(seed)

    negative_pairs = []

    img_ids, filenames, captions = extract_columns(split_data)

    num_images = len(img_ids)

    for image_idx, (img_id, filename, caption_list) in enumerate(
        zip(img_ids, filenames, captions)
    ):
        for _ in caption_list:
            negative_image_idx = rng.randrange(num_images)

            while negative_image_idx == image_idx:
                negative_image_idx = rng.randrange(num_images)

            negative_caption = rng.choice(captions[negative_image_idx])

            negative_pairs.append({
                "img_id": img_id,
                "filename": filename,
                "caption": negative_caption,
                "label": 0,
            })

    return pd.DataFrame(negative_pairs)

def create_binary_dataset(positive_pairs, negative_pairs, seed=42):
    """
    Combine positive and negative image-caption pairs into one binary dataset.

    Args:
        positive_pairs: DataFrame containing matching image-caption pairs.
        negative_pairs: DataFrame containing non-matching image-caption pairs.
        seed: Random seed used for reproducible shuffling.

    Returns:
        A shuffled pandas DataFrame containing both positive and negative pairs.
    """
    binary_dataset = pd.concat(
        [positive_pairs, negative_pairs],
        ignore_index=True,
    )

    binary_dataset = binary_dataset.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return binary_dataset