import random
import matplotlib.pyplot as plt


def show_image_to_text_retrieval_examples(
    hf_dataset,
    similarity_matrix,
    n_examples=3,
    top_k=10,
    random_state=42,
    multi_caption=False,
    candidate_captions=None,
    sample_indices=None,
):
    """
    Display image-to-text retrieval examples.

    Args:
        hf_dataset: Validation dataset.
        similarity_matrix: Image-to-text similarity matrix.
        n_examples: Number of random examples to display when
            sample_indices is not provided.
        top_k: Number of retrieved captions to display.
        random_state: Seed for reproducible sampling.
        multi_caption: Whether multiple ground-truth captions are
            considered correct.
        candidate_captions: Optional list of candidate captions
            corresponding to similarity_matrix columns.
        sample_indices: Optional list of image indices to visualize.
            If provided, random sampling is skipped.
    """
    if sample_indices is None:
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
        for i, caption in enumerate(true_captions, start=1):
            print(f"{i}. {caption}")

        print("\nTop retrieved captions:")

        matched_caption_numbers = set()

        for rank, idx in enumerate(top_indices, start=1):
            if candidate_captions is not None:
                retrieved_caption = candidate_captions[idx]
            else:
                retrieved_caption = hf_dataset[idx]["caption"][0]

            score = similarities[idx].item()

            marker = ""

            if multi_caption:
                if retrieved_caption in true_captions:
                    caption_number = true_captions.index(retrieved_caption) + 1
                    matched_caption_numbers.add(caption_number)
                    marker = f"✅ Ground truth #{caption_number}"
            else:
                marker = "✅" if idx == sample_idx else ""

            print(
                f"{rank}. {retrieved_caption} "
                f"| score={score:.3f} {marker}"
            )

        if multi_caption:
            print(
                f"\nRetrieved {len(matched_caption_numbers)} "
                f"of {len(true_captions)} ground-truth captions "
                f"in top {top_k}."
            )

        print("-" * 100)


def show_text_to_image_retrieval_examples(
    hf_dataset,
    similarity_matrix,
    n_examples=3,
    top_k=5,
    random_state=42,
    sample_indices=None,
):
    rng = random.Random(random_state)

    if sample_indices is None:
        sample_indices = rng.sample(
            range(len(hf_dataset)),
            k=min(n_examples, len(hf_dataset)),
        )

    for sample_idx in sample_indices:
        query_caption = hf_dataset[sample_idx]["caption"][0]

        similarities = similarity_matrix[sample_idx]
        top_indices = similarities.topk(top_k).indices.tolist()

        print(f"Query caption:")
        print(query_caption)

        fig, axes = plt.subplots(1, top_k, figsize=(4 * top_k, 4))

        if top_k == 1:
            axes = [axes]

        for rank, (ax, idx) in enumerate(zip(axes, top_indices), start=1):
            image = hf_dataset[idx]["image"]

            marker = "✅" if idx == sample_idx else ""

            ax.imshow(image)
            ax.axis("off")
            ax.set_title(
                f"Rank {rank} {marker}\nscore={similarities[idx]:.3f}",
                fontsize=10,
            )

        plt.show()
        print("-" * 100)