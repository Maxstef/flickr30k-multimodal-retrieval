import random
import matplotlib.pyplot as plt


def show_sample(dataset, idx):
    sample = dataset["test"][idx]

    plt.figure(figsize=(8, 8))
    plt.imshow(sample["image"])
    plt.axis("off")
    plt.show()

    print(f"Index: {idx}")
    print(f"Image ID: {sample['img_id']}")
    print(f"Filename: {sample['filename']}\n")

    for i, caption in enumerate(sample["caption"], start=1):
        print(f"{i}. {caption}")


def get_random_indices(dataset, n=3, seed=42):
    random.seed(seed)
    return random.sample(range(len(dataset["test"])), n)