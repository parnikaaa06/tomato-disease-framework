"""Classification metrics for predictions collected over a complete split."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    targets: np.ndarray | list[int],
    predictions: np.ndarray | list[int],
    class_names: list[str],
) -> dict:
    labels = list(range(len(class_names)))
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).tolist(),
    }
