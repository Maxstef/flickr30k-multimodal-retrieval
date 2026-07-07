from pathlib import Path
import sys
import shutil
import json

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from datasets import concatenate_datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DATASET_NAME,
    DEVICE,
    MODELS_DIR,
    MAX_CAPTION_LENGTH,
    EMBEDDING_DIM,
    IMAGE_FEATURE_DIM,
    PROJECTION_DIM,
    RANDOM_SEED,
)
from src.data.loaders import load_flickr30k_splits
from src.text.tokenization import build_vocab, encode_caption
from src.features.image_features import (
    get_resnet18_feature_extractor,
    get_resnet18_transforms,
    extract_image_features,
)
from src.models.mini_clip import MiniCLIP


APP_DATA_DIR = PROJECT_ROOT / "app_data"
APP_IMAGES_DIR = APP_DATA_DIR / "images"

NUM_IMAGES = 1000
RANDOM_SEED = 42
IMAGE_SIZE = (384, 384)
JPEG_QUALITY = 85


def save_resized_image(image, output_path):
    image = image.convert("RGB")
    image.thumbnail(IMAGE_SIZE)
    image.save(output_path, format="JPEG", quality=JPEG_QUALITY)


def main():
    APP_DATA_DIR.mkdir(exist_ok=True)
    APP_IMAGES_DIR.mkdir(exist_ok=True)

    train_data, val_data, test_data = load_flickr30k_splits(DATASET_NAME)

    full_dataset = concatenate_datasets([
        train_data,
        val_data,
        test_data,
    ])

    demo_data = full_dataset.shuffle(seed=RANDOM_SEED).select(
        range(NUM_IMAGES)
    )

    vocab = build_vocab(train_data["caption"], min_freq=1)

    model = MiniCLIP(
        vocab_size=len(vocab),
        text_embedding_dim=EMBEDDING_DIM,
        image_feature_dim=IMAGE_FEATURE_DIM,
        projection_dim=PROJECTION_DIM,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "mini_clip.pt",
            map_location=DEVICE,
        )
    )

    model.eval()

    # save vocab
    with open(APP_DATA_DIR / "vocab.json", "w") as f:
            json.dump(vocab, f)

    image_feature_extractor = get_resnet18_feature_extractor(DEVICE)
    image_transform = get_resnet18_transforms()

    image_rows = []
    caption_rows = []

    print("Saving demo images and metadata...")

    for image_idx, sample in enumerate(tqdm(demo_data)):
        original_filename = sample["filename"]
        output_filename = f"{image_idx:05d}_{original_filename}"
        output_path = APP_IMAGES_DIR / output_filename

        save_resized_image(sample["image"], output_path)

        image_rows.append(
            {
                "image_idx": image_idx,
                "filename": output_filename,
            }
        )

        for caption in sample["caption"]:
            caption_rows.append(
                {
                    "caption_idx": len(caption_rows),
                    "caption": caption,
                }
            )

    images_df = pd.DataFrame(image_rows)
    captions_df = pd.DataFrame(caption_rows)

    images_df.to_csv(APP_DATA_DIR / "images.csv", index=False)
    captions_df.to_csv(APP_DATA_DIR / "captions.csv", index=False)

    print("Extracting image features...")

    image_features = extract_image_features(
        demo_data,
        image_feature_extractor,
        image_transform,
        DEVICE,
    )

    ordered_image_features = []

    for sample in demo_data:
        ordered_image_features.append(image_features[sample["filename"]])

    ordered_image_features = torch.stack(ordered_image_features).to(DEVICE)

    print("Projecting image embeddings with Mini-CLIP...")

    with torch.inference_mode():
        image_embeddings = model.image_projection(ordered_image_features)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=1)

    torch.save(
        image_embeddings.cpu(),
        APP_DATA_DIR / "image_embeddings.pt",
    )

    print("Encoding caption embeddings...")

    caption_tensors = []

    for caption in captions_df["caption"]:
        caption_ids = encode_caption(
            caption,
            vocab,
            max_length=MAX_CAPTION_LENGTH,
        )
        caption_tensors.append(torch.tensor(caption_ids, dtype=torch.long))

    caption_tensors = torch.stack(caption_tensors)

    caption_embeddings = []

    with torch.inference_mode():
        for start_idx in range(0, len(caption_tensors), 128):
            batch = caption_tensors[start_idx:start_idx + 128].to(DEVICE)

            dummy_image_features = torch.zeros(
                batch.shape[0],
                IMAGE_FEATURE_DIM,
                device=DEVICE,
            )

            _, text_embeddings = model(dummy_image_features, batch)
            caption_embeddings.append(text_embeddings.cpu())

    caption_embeddings = torch.cat(caption_embeddings)

    torch.save(
        caption_embeddings,
        APP_DATA_DIR / "caption_embeddings.pt",
    )

    print("App data prepared successfully.")
    print(f"Images: {len(images_df):,}")
    print(f"Captions: {len(captions_df):,}")
    print(f"Image embeddings: {image_embeddings.shape}")
    print(f"Caption embeddings: {caption_embeddings.shape}")


if __name__ == "__main__":
    main()