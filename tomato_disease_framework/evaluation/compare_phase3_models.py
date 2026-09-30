"""Compare saved Phase 3 PlantVillage test metrics without rerunning inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


BASELINE_RESULTS_PATH = Path(
    "results/phase3/efficientnet_b0_baseline/test_results.yaml"
)
ATTENTION_RESULTS_PATH = Path(
    "results/phase3/efficientnet_b0_dynamic_attention/test_results.yaml"
)
COMPARISON_OUTPUT_PATH = Path("results/phase3/phase3_model_comparison.yaml")
OFFICIAL_TEST_SAMPLE_COUNT = 2725
METRIC_KEYS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


def _load_results(path: Path, model_label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{model_label} test results not found: {path}. "
            "Run the corresponding official test evaluation first."
        )
    with path.open("r", encoding="utf-8") as results_file:
        results = yaml.safe_load(results_file)
    if not isinstance(results, dict):
        raise ValueError(f"{model_label} test results must contain a YAML mapping: {path}")
    return results


def _validate_results(
    results: dict[str, Any],
    model_label: str,
) -> tuple[list[str], dict[str, Any], list[list[int]]]:
    class_names = results.get("class_names")
    metrics = results.get("metrics")
    sample_count = results.get("test_sample_count")

    if sample_count != OFFICIAL_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"{model_label} result sample count is {sample_count!r}; "
            f"expected {OFFICIAL_TEST_SAMPLE_COUNT}."
        )
    if not isinstance(class_names, list) or len(class_names) != 10:
        raise ValueError(f"{model_label} results must list exactly 10 class names.")
    if any(not isinstance(name, str) for name in class_names):
        raise ValueError(f"{model_label} class names must all be strings.")
    if not isinstance(metrics, dict):
        raise ValueError(f"{model_label} results must contain a metrics mapping.")
    missing_metrics = [key for key in METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise ValueError(
            f"{model_label} results are missing metric(s): {', '.join(missing_metrics)}."
        )
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError(f"{model_label} results must contain per-class metrics.")
    missing_classes = [name for name in class_names if name not in per_class]
    if missing_classes:
        raise ValueError(
            f"{model_label} per-class metrics are missing: {', '.join(missing_classes)}."
        )
    for name in class_names:
        if "f1" not in per_class[name]:
            raise ValueError(f"{model_label} per-class metrics lack F1 for {name}.")

    matrix = metrics.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(class_names)
        or any(not isinstance(row, list) or len(row) != len(class_names) for row in matrix)
    ):
        raise ValueError(
            f"{model_label} confusion matrix must be {len(class_names)}x{len(class_names)}."
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for row in matrix
        for value in row
    ):
        raise ValueError(f"{model_label} confusion matrix must contain nonnegative integers.")
    if sum(map(sum, matrix)) != OFFICIAL_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"{model_label} confusion matrix total must equal "
            f"{OFFICIAL_TEST_SAMPLE_COUNT}."
        )

    return class_names, metrics, matrix


def build_comparison(
    baseline_results: dict[str, Any],
    attention_results: dict[str, Any],
) -> dict[str, Any]:
    """Validate and combine saved model results using the fixed class order."""
    baseline_classes, baseline_metrics, baseline_matrix = _validate_results(
        baseline_results, "Baseline"
    )
    attention_classes, attention_metrics, attention_matrix = _validate_results(
        attention_results, "Attention"
    )
    if baseline_classes != attention_classes:
        raise ValueError(
            "Class order mismatch between baseline and attention test results; "
            "cannot compare per-class metrics or confusion matrices."
        )

    aggregate = {
        key: {
            "baseline": float(baseline_metrics[key]),
            "attention": float(attention_metrics[key]),
            "attention_minus_baseline": (
                float(attention_metrics[key]) - float(baseline_metrics[key])
            ),
        }
        for key in METRIC_KEYS
    }
    per_class_f1 = {
        class_name: {
            "baseline_f1": float(baseline_metrics["per_class"][class_name]["f1"]),
            "attention_f1": float(attention_metrics["per_class"][class_name]["f1"]),
            "attention_minus_baseline": (
                float(attention_metrics["per_class"][class_name]["f1"])
                - float(baseline_metrics["per_class"][class_name]["f1"])
            ),
        }
        for class_name in baseline_classes
    }

    confusion_changes = []
    row_prediction_count_changes = []
    for actual_index, actual_class in enumerate(baseline_classes):
        for predicted_index, predicted_class in enumerate(baseline_classes):
            baseline_count = baseline_matrix[actual_index][predicted_index]
            attention_count = attention_matrix[actual_index][predicted_index]
            difference = attention_count - baseline_count
            if difference:
                confusion_changes.append(
                    {
                        "true_class": actual_class,
                        "predicted_class": predicted_class,
                        "baseline_count": baseline_count,
                        "attention_count": attention_count,
                        "attention_minus_baseline": difference,
                    }
                )

        baseline_row = baseline_matrix[actual_index]
        attention_row = attention_matrix[actual_index]
        minimum_changed_predictions = sum(
            max(0, baseline_row[index] - attention_row[index])
            for index in range(len(baseline_classes))
        )
        row_prediction_count_changes.append(
            {
                "true_class": actual_class,
                "baseline_correct": baseline_row[actual_index],
                "attention_correct": attention_row[actual_index],
                "attention_minus_baseline_correct": (
                    attention_row[actual_index] - baseline_row[actual_index]
                ),
                "minimum_changed_predictions_inferred_from_counts": (
                    minimum_changed_predictions
                ),
            }
        )

    baseline_misclassified = sum(
        count
        for row_index, row in enumerate(baseline_matrix)
        for column_index, count in enumerate(row)
        if row_index != column_index
    )
    attention_misclassified = sum(
        count
        for row_index, row in enumerate(attention_matrix)
        for column_index, count in enumerate(row)
        if row_index != column_index
    )

    return {
        "dataset_context": "Controlled PlantVillage official test-set results",
        "test_sample_count": OFFICIAL_TEST_SAMPLE_COUNT,
        "class_names_in_matrix_order": baseline_classes,
        "aggregate_metrics": aggregate,
        "per_class_f1": per_class_f1,
        "confusion_matrix_comparison": {
            "baseline_matrix": baseline_matrix,
            "attention_matrix": attention_matrix,
            "misclassified_sample_counts": {
                "baseline": baseline_misclassified,
                "attention": attention_misclassified,
                "attention_minus_baseline": (
                    attention_misclassified - baseline_misclassified
                ),
            },
            "cell_count_changes": confusion_changes,
            "per_true_class_summary": row_prediction_count_changes,
            "interpretation_note": (
                "Confusion matrices provide aggregate counts, not sample identities. "
                "The minimum changed-prediction counts are inferred from per-class "
                "prediction-count differences and do not identify which images changed."
            ),
        },
    }


def print_comparison(comparison: dict[str, Any]) -> None:
    print(comparison["dataset_context"])
    print(f"Official test samples: {comparison['test_sample_count']}")
    print()
    print(f"{'Metric':<20}{'Baseline':>14}{'Attention':>14}{'Attention - Baseline':>24}")
    for name, values in comparison["aggregate_metrics"].items():
        print(
            f"{name:<20}{values['baseline']:>14.6f}"
            f"{values['attention']:>14.6f}"
            f"{values['attention_minus_baseline']:>24.6f}"
        )

    print("\nPer-class F1")
    print(f"{'Class':<42}{'Baseline':>12}{'Attention':>12}{'Difference':>14}")
    for class_name, values in comparison["per_class_f1"].items():
        print(
            f"{class_name:<42}{values['baseline_f1']:>12.6f}"
            f"{values['attention_f1']:>12.6f}"
            f"{values['attention_minus_baseline']:>14.6f}"
        )

    changes = comparison["confusion_matrix_comparison"]["cell_count_changes"]
    errors = comparison["confusion_matrix_comparison"]["misclassified_sample_counts"]
    print(
        "\nMisclassified sample counts (aggregate): "
        f"baseline={errors['baseline']}, attention={errors['attention']}, "
        f"attention - baseline={errors['attention_minus_baseline']}"
    )
    print(f"\nConfusion-matrix cells with changed counts: {len(changes)}")
    print(f"{'True class':<34}{'Predicted class':<34}{'Baseline':>10}{'Attention':>12}{'Delta':>9}")
    for change in changes:
        print(
            f"{change['true_class']:<34}{change['predicted_class']:<34}"
            f"{change['baseline_count']:>10}{change['attention_count']:>12}"
            f"{change['attention_minus_baseline']:>9}"
        )
    print(
        "\nAggregate confusion-matrix counts do not identify individual changed images."
    )


def main() -> None:
    baseline = _load_results(BASELINE_RESULTS_PATH, "Baseline")
    attention = _load_results(ATTENTION_RESULTS_PATH, "Attention")
    comparison = build_comparison(baseline, attention)

    COMPARISON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(
            comparison,
            output_file,
            sort_keys=False,
            allow_unicode=True,
        )
    print_comparison(comparison)
    print(f"\nComparison saved to: {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
