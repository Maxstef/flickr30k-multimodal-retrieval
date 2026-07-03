import matplotlib.pyplot as plt


def get_image_by_filename(hf_dataset, filename):
    """
    Retrieve a PIL image from a Hugging Face dataset using its filename.
    """
    filename_to_index = {
        sample_filename: idx
        for idx, sample_filename in enumerate(hf_dataset["filename"])
    }

    image_idx = filename_to_index[filename]
    return hf_dataset[image_idx]["image"]


def format_label(label):
    """
    Convert a binary label into a readable text label.
    """
    return "Positive match" if label == 1 else "Negative pair"


def format_prediction(prediction, probability):
    """
    Format model prediction and probability for display.
    """
    label = "Positive" if prediction == 1 else "Negative"
    return f"{label} ({probability:.3f})"


def show_prediction_examples(
    dataframe,
    hf_dataset,
    category=None,
    n=5,
    random_state=42,
    title=None,
):
    """
    Display random image-caption examples with predictions from both models.

    Args:
        dataframe: DataFrame containing captions, labels, predictions, probabilities,
            filenames, and prediction categories.
        hf_dataset: Hugging Face dataset split containing original images.
        category: Optional prediction category to filter by.
        n: Number of examples to display.
        random_state: Random seed for reproducible sampling.
        title: Optional section title printed above the examples.
    """
    if category is not None:
        examples = dataframe[
            dataframe["prediction_category"] == category
        ]
    else:
        examples = dataframe

    if len(examples) == 0:
        print("No examples found.")
        return

    examples = examples.sample(
        n=min(n, len(examples)),
        random_state=random_state,
    )

    if title is not None:
        print(title)

    for _, row in examples.iterrows():
        image = get_image_by_filename(
            hf_dataset,
            row["filename"],
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

        print(f"Filename: {row['filename']}")
        print(f"Caption: {row['caption']}")
        print(f"Ground truth: {format_label(row['label'])}")
        print(
            "Frozen model: "
            f"{format_prediction(row['frozen_pred'], row['frozen_prob'])}"
        )
        print(
            "Fine-tuned model: "
            f"{format_prediction(row['finetuned_pred'], row['finetuned_prob'])}"
        )

        if "prediction_category" in row:
            print(f"Category: {row['prediction_category']}")

        print("-" * 80)