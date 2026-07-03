import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_thresholds(probabilities, labels, thresholds):
    """
    Evaluate binary classification metrics for multiple probability thresholds.

    Args:
        probabilities: Predicted probabilities for the positive class.
        labels: Ground-truth binary labels.
        thresholds: Iterable of threshold values.

    Returns:
        DataFrame with accuracy, precision, recall, and F1-score per threshold.
    """
    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(labels, predictions),
                "precision": precision_score(labels, predictions, zero_division=0),
                "recall": recall_score(labels, predictions, zero_division=0),
                "f1": f1_score(labels, predictions, zero_division=0),
            }
        )

    return pd.DataFrame(rows)