"""Scoring functions for Litmus benchmark."""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def score_regression(
    targets: list[float],
    predictions: list[float],
) -> dict[str, float]:
    """Score regression predictions.

    Parameters
    ----------
    targets : list[float]
        Ground truth values.
    predictions : list[float]
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: spearman, pearson, r2, rmse, mae.
    """
    targets_arr = np.array(targets)
    preds_arr = np.array(predictions)

    spearman_r, _ = stats.spearmanr(targets_arr, preds_arr)
    pearson_r, _ = stats.pearsonr(targets_arr, preds_arr)

    return {
        "spearman": float(spearman_r),
        "pearson": float(pearson_r),
        "r2": float(r2_score(targets_arr, preds_arr)),
        "rmse": float(np.sqrt(mean_squared_error(targets_arr, preds_arr))),
        "mae": float(mean_absolute_error(targets_arr, preds_arr)),
    }


def score_classification(
    targets: list[str],
    predictions: list[str],
    task_type: str,
    choices: list[str] | None = None,
) -> dict[str, float]:
    """Score classification predictions.

    Parameters
    ----------
    targets : list[str]
        Ground truth labels.
    predictions : list[str]
        Predicted labels.
    task_type : str
        One of "binary" or "multiclass".
    choices : list[str] | None
        Ordered list of class labels (needed for AUROC in binary).

    Returns
    -------
    dict[str, float]
        Dictionary with keys: accuracy, f1_macro, f1_weighted.
        For binary tasks, also includes auroc.
    """
    result = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
    }

    if task_type == "binary" and choices and len(choices) == 2:
        try:
            target_binary = [1 if t == choices[0] else 0 for t in targets]
            pred_binary = [1 if p == choices[0] else 0 for p in predictions]
            result["auroc"] = float(roc_auc_score(target_binary, pred_binary))
        except ValueError:
            pass

    return result


def score_multilabel(
    targets: list[list[str]],
    predictions: list[list[str]],
    all_labels: list[str],
) -> dict[str, float]:
    """Score multilabel predictions.

    Parameters
    ----------
    targets : list[list[str]]
        Ground truth label sets.
    predictions : list[list[str]]
        Predicted label sets.
    all_labels : list[str]
        Complete list of possible labels.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: exact_match, f1_macro.
    """
    def to_binary(label_lists: list[list[str]]) -> np.ndarray:
        matrix = np.zeros((len(label_lists), len(all_labels)), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        for i, labels in enumerate(label_lists):
            for label in labels:
                if label in label_to_idx:
                    matrix[i, label_to_idx[label]] = 1
        return matrix

    targets_bin = to_binary(targets)
    preds_bin = to_binary(predictions)

    exact_match = float(np.mean(np.all(targets_bin == preds_bin, axis=1)))

    # Per-label F1, then macro average
    per_label_f1 = []
    for j in range(len(all_labels)):
        f1 = f1_score(targets_bin[:, j], preds_bin[:, j], zero_division=0)
        per_label_f1.append(f1)

    return {
        "exact_match": exact_match,
        "f1_macro": float(np.mean(per_label_f1)),
    }
