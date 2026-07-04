import random
import matplotlib.pyplot as plt


def show_image_to_text_retrieval_examples(
    hf_dataset,
    similarity_matrix,
    n_examples=3,
    top_k=10,
    random_state=42,
):
    """
    Display random image-to-text retrieval examples.

    Assumes the text embedding for each validation image was created
    from the first caption of that image.
    """
    rng = random.Random(random_state)

    sample_indices = rng.sample(
        range(len(hf_dataset)),
        k=min(n_examples, len(hf_dataset)),
    )

    for sample_idx in sample_indices:
        sample = hf_dataset[sample_idx]
        image = sample["image"]
        true_captions = sample["caption"]

        similarities = similarity_matrix[sample_idx]
        top_indices = similarities.topk(top_k).indices.tolist()

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

        print(f"Image index: {sample_idx}")
        print("\nGround-truth captions:")
        for caption in true_captions:
            print(f"- {caption}")

        print("\nTop retrieved captions:")
        for rank, idx in enumerate(top_indices, start=1):
            retrieved_caption = hf_dataset[idx]["caption"][0]
            score = similarities[idx].item()
            marker = "✅" if idx == sample_idx else ""

            print(
                f"{rank}. {retrieved_caption} "
                f"| score={score:.3f} {marker}"
            )

        print("-" * 100)