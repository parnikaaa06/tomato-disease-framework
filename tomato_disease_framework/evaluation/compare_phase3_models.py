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
PER_CLASS_METRIC_KEYS = ("precision", "recall", "f1")


def _load_results(path: Path, model_label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{model_label} test results not found: {path}. "
            "Copy the saved test_results.yaml into the project results directory."
        )
    with path.open("r", encoding="utf-8") as results_file:
        results = yaml.safe_load(results_file)
    if not isinstance(results, dict):
        raise ValueError(f"{model_label} test results must contain a YAML mapping: {path}")
    return results


def _validate_results(
    results: dict[str, Any],
    model_label: str,
) -> tuple[list[str], dict[str, Any], list[list[int]], str]:
    class_names = results.get("class_names")
    metrics = results.get("metrics")
    sample_count = results.get("test_sample_count")
    split_path = results.get("test_split")

    if sample_count != OFFICIAL_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"{model_label} result sample count is {sample_count!r}; "
            f"expected {OFFICIAL_TEST_SAMPLE_COUNT}."
        )
    if not isinstance(split_path, str) or not split_path:
        raise ValueError(f"{model_label} results must identify the test split.")
    if not isinstance(class_names, list) or len(class_names) != 10:
        raise ValueError(f"{model_label} results must list exactly 10 class names.")
    if any(not isinstance(name, str) or not name for name in class_names):
        raise ValueError(f"{model_label} class names must all be non-empty strings.")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"{model_label} class list contains duplicates.")
    if not isinstance(metrics, dict):
        raise ValueError(f"{model_label} results must contain a metrics mapping.")

    for key in METRIC_KEYS:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{model_label} results require a numeric '{key}' metric."
            )

    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError(f"{model_label} results must contain per-class metrics.")
    for class_name in class_names:
        class_metrics = per_class.get(class_name)
        if not isinstance(class_metrics, dict):
            raise ValueError(
                f"{model_label} per-class metrics are missing for {class_name}."
            )
        for key in (*PER_CLASS_METRIC_KEYS, "support"):
            value = class_metrics.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{model_label} per-class '{key}' is missing or non-numeric "
                    f"for {class_name}."
                )

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
        raise ValueError(
            f"{model_label} confusion matrix must contain nonnegative integers."
        )
    if sum(map(sum, matrix)) != OFFICIAL_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"{model_label} confusion matrix total must equal "
            f"{OFFICIAL_TEST_SAMPLE_COUNT}."
        )
    if [
        sum(row)
        for row in matrix
    ] != [int(per_class[name]["support"]) for name in class_names]:
        raise ValueError(
            f"{model_label} per-class support values do not match confusion-matrix row totals."
        )

    return class_names, metrics, matrix, split_path


def _count_errors(matrix: list[list[int]]) -> dict[str, int]:
    correct = sum(matrix[index][index] for index in range(len(matrix)))
    return {
        "correct": correct,
        "incorrect": OFFICIAL_TEST_SAMPLE_COUNT - correct,
        "total_errors": OFFICIAL_TEST_SAMPLE_COUNT - correct,
    }


def _misclassifications(
    matrix: list[list[int]],
    class_names: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "true_class": class_names[true_index],
            "predicted_class": class_names[predicted_index],
            "count": count,
        }
        for true_index, row in enumerate(matrix)
        for predicted_index, count in enumerate(row)
        if true_index != predicted_index and count > 0
    ]


def build_comparison(
    baseline_results: dict[str, Any],
    attention_results: dict[str, Any],
) -> dict[str, Any]:
    """Validate and combine saved model results using the fixed class order."""
    baseline_classes, baseline_metrics, baseline_matrix, baseline_split = (
        _validate_results(baseline_results, "Baseline")
    )
    attention_classes, attention_metrics, attention_matrix, attention_split = (
        _validate_results(attention_results, "Dynamic attention")
    )
    if baseline_classes != attention_classes:
        raise ValueError(
            "Class order mismatch between baseline and dynamic-attention "
            "results; per-class metrics and confusion matrices cannot be compared."
        )
    if baseline_split != attention_split:
        raise ValueError(
            f"Test split mismatch: baseline uses {baseline_split!r}, "
            f"dynamic attention uses {attention_split!r}."
        )

    aggregate_baseline = {
        key: float(baseline_metrics[key]) for key in METRIC_KEYS
    }
    aggregate_attention = {
        key: float(attention_metrics[key]) for key in METRIC_KEYS
    }
    aggregate_delta = {
        key: aggregate_attention[key] - aggregate_baseline[key]
        for key in METRIC_KEYS
    }

    per_class_metrics: dict[str, Any] = {}
    for class_name in baseline_classes:
        baseline_class = baseline_metrics["per_class"][class_name]
        attention_class = attention_metrics["per_class"][class_name]
        per_class_metrics[class_name] = {
            "baseline": {
                **{
                    key: float(baseline_class[key])
                    for key in PER_CLASS_METRIC_KEYS
                },
                "support": int(baseline_class["support"]),
            },
            "dynamic_attention": {
                **{
                    key: float(attention_class[key])
                    for key in PER_CLASS_METRIC_KEYS
                },
                "support": int(attention_class["support"]),
            },
            "delta": {
                key: (
                    float(attention_class[key]) - float(baseline_class[key])
                )
                for key in PER_CLASS_METRIC_KEYS
            },
        }
        if int(baseline_class["support"]) != int(attention_class["support"]):
            raise ValueError(
                f"Per-class support differs for {class_name}: "
                f"baseline={baseline_class['support']}, "
                f"dynamic attention={attention_class['support']}."
            )

    baseline_error_counts = _count_errors(baseline_matrix)
    attention_error_counts = _count_errors(attention_matrix)
    misclassification_pairs = []
    baseline_pairs = {
        (item["true_class"], item["predicted_class"]): item["count"]
        for item in _misclassifications(baseline_matrix, baseline_classes)
    }
    attention_pairs = {
        (item["true_class"], item["predicted_class"]): item["count"]
        for item in _misclassifications(attention_matrix, attention_classes)
    }
    for true_class, predicted_class in sorted(
        set(baseline_pairs) | set(attention_pairs)
    ):
        baseline_count = baseline_pairs.get((true_class, predicted_class), 0)
        attention_count = attention_pairs.get((true_class, predicted_class), 0)
        misclassification_pairs.append(
            {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "baseline_count": baseline_count,
                "dynamic_attention_count": attention_count,
                "delta": attention_count - baseline_count,
            }
        )

    error_delta = {
        key: attention_error_counts[key] - baseline_error_counts[key]
        for key in ("correct", "incorrect")
    }
    return {
        "dataset_context": (
            "Controlled PlantVillage test-set results; these results do not "
            "establish real-world field reliability."
        ),
        "experiment": {
            "baseline": {
                "name": baseline_results.get("model_name", "EfficientNetB0"),
                "results_file": str(BASELINE_RESULTS_PATH),
            },
            "proposed": {
                "name": attention_results.get(
                    "model_name", "EfficientNetB0DynamicAttention"
                ),
                "results_file": str(ATTENTION_RESULTS_PATH),
            },
        },
        "dataset": {
            "test_samples": OFFICIAL_TEST_SAMPLE_COUNT,
            "test_split": baseline_split,
            "class_count": len(baseline_classes),
            "classes": baseline_classes,
        },
        "aggregate_metrics": {
            "baseline": aggregate_baseline,
            "dynamic_attention": aggregate_attention,
            "delta": aggregate_delta,
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "class_order": baseline_classes,
            "baseline": baseline_matrix,
            "dynamic_attention": attention_matrix,
        },
        "error_analysis": {
            "baseline": {
                **baseline_error_counts,
                "misclassifications": _misclassifications(
                    baseline_matrix, baseline_classes
                ),
            },
            "dynamic_attention": {
                **attention_error_counts,
                "misclassifications": _misclassifications(
                    attention_matrix, attention_classes
                ),
            },
            "delta": error_delta,
            "misclassification_pair_comparison": misclassification_pairs,
        },
        "comparison_notes": [
            "Both result files report the same official test split and sample count.",
            "Class ordering is validated before per-class and confusion-matrix comparisons.",
            "Metric deltas are calculated as dynamic attention minus baseline.",
            "Misclassification entries describe aggregate confusion-matrix counts, not image identities.",
            "These controlled PlantVillage test-set results do not establish real-world field reliability.",
        ],
    }


def print_comparison(comparison: dict[str, Any]) -> None:
    aggregate = comparison["aggregate_metrics"]
    print(comparison["dataset_context"])
    print(f"Official test samples: {comparison['dataset']['test_samples']}\n")
    print(
        f"{'Metric':<24}{'Baseline':>14}"
        f"{'Dynamic Attention':>20}{'Delta':>14}"
    )
    for key, label in (
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ):
        print(
            f"{label:<24}{aggregate['baseline'][key]:>14.6f}"
            f"{aggregate['dynamic_attention'][key]:>20.6f}"
            f"{aggregate['delta'][key]:>14.6f}"
        )

    print("\nPer-class F1")
    print(f"{'Class':<48}{'Baseline':>12}{'Attention':>12}{'Delta':>12}")
    for class_name, metrics in comparison["per_class_metrics"].items():
        print(
            f"{class_name:<48}{metrics['baseline']['f1']:>12.6f}"
            f"{metrics['dynamic_attention']['f1']:>12.6f}"
            f"{metrics['delta']['f1']:>12.6f}"
        )

    errors = comparison["error_analysis"]
    print(
        "\nTest errors: "
        f"baseline={errors['baseline']['incorrect']}, "
        f"dynamic attention={errors['dynamic_attention']['incorrect']}"
    )


def main() -> None:
    baseline_results = _load_results(BASELINE_RESULTS_PATH, "Baseline")
    attention_results = _load_results(
        ATTENTION_RESULTS_PATH, "Dynamic attention"
    )
    comparison = build_comparison(baseline_results, attention_results)

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
