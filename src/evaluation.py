"""
evaluation.py — Metrics computation and results export for vcfp_reproduction.

Computes: Accuracy, Precision, Recall, F1 (macro/weighted),
          Confusion matrices, Semantic Distance, Normalized Semantic Distance.
Exports  : CSV tables to results/tables/.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.semantic import evaluate_semantic_metrics, cosine_similarity


# ── Per-Experiment Evaluation ──────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all classification metrics for a CV experiment.

    Returns a flat dict of metric name → float value.
    """
    metrics = {
        "accuracy":           accuracy_score(y_true, y_pred),
        "precision_macro":    precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":       recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return metrics


def compute_semantic_metrics_for_cv(
    all_true_int: np.ndarray,
    all_pred_int: np.ndarray,
    inv_label_map: Dict[int, str],
    vec_dict: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """
    Convert integer labels to command strings and compute semantic metrics.

    Returns: (mean_semantic_distance, mean_normalized_distance)
    """
    true_cmds = [inv_label_map[i] for i in all_true_int]
    pred_cmds = [inv_label_map[i] for i in all_pred_int]
    return evaluate_semantic_metrics(true_cmds, pred_cmds, vec_dict)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def full_evaluation(
    model_name: str,
    cv_results: Dict,
    inv_label_map: Dict[int, str],
    vec_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    """
    Run full evaluation pipeline for one model's CV results.

    Args:
        model_name   : display name for the model
        cv_results   : output of one of the cross_validate_* functions
        inv_label_map: int → command string mapping
        vec_dict     : semantic vectors (None to skip semantic metrics)

    Returns dict with all computed metrics.
    """
    y_true = cv_results["all_true"]
    y_pred = cv_results["all_pred"]

    metrics = compute_classification_metrics(y_true, y_pred)

    # Per-fold statistics from CV results
    metrics["fold_accuracies"] = cv_results.get("fold_accuracies", [])
    metrics["std_accuracy"]    = cv_results.get("std_accuracy", 0.0)

    # Semantic metrics (if vectors available)
    if vec_dict is not None:
        mean_sd, mean_nd = compute_semantic_metrics_for_cv(
            y_true, y_pred, inv_label_map, vec_dict
        )
        metrics["semantic_distance"]            = mean_sd
        metrics["normalized_semantic_distance"] = mean_nd
    else:
        metrics["semantic_distance"]            = None
        metrics["normalized_semantic_distance"] = None

    # Confusion matrix stored as numpy array
    metrics["confusion_matrix"] = get_confusion_matrix(y_true, y_pred)
    metrics["model_name"] = model_name

    return metrics


# ── Results Aggregation & Export ───────────────────────────────────────────

SCALAR_METRICS = [
    "accuracy", "std_accuracy",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "semantic_distance", "normalized_semantic_distance",
]

PAPER_REFERENCE = {
    "LL-Jaccard":       {"accuracy": 0.174, "semantic_distance": 0.949, "normalized_semantic_distance": 46.99},
    "LL-NB":            {"accuracy": 0.338, "semantic_distance": 0.955, "normalized_semantic_distance": 34.11},
    "VNG++":            {"accuracy": 0.249, "semantic_distance": 0.950, "normalized_semantic_distance": 43.80},
    "P-SVM (AdaBoost)": {"accuracy": 0.334, "semantic_distance": 0.956, "normalized_semantic_distance": 37.68},
}


def build_results_dataframe(all_eval_results: List[Dict]) -> pd.DataFrame:
    """
    Build a clean comparison DataFrame from a list of evaluation result dicts.
    """
    rows = []
    for res in all_eval_results:
        row = {"Model": res["model_name"]}
        for m in SCALAR_METRICS:
            val = res.get(m)
            if isinstance(val, float):
                row[m] = round(val, 4)
            else:
                row[m] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    # Rename for readability
    df.rename(columns={
        "accuracy":                      "Accuracy",
        "std_accuracy":                  "Acc Std",
        "precision_macro":               "Precision",
        "recall_macro":                  "Recall",
        "f1_macro":                      "F1",
        "precision_weighted":            "Precision (W)",
        "recall_weighted":               "Recall (W)",
        "f1_weighted":                   "F1 (W)",
        "semantic_distance":             "Semantic Dist",
        "normalized_semantic_distance":  "Norm Sem Dist",
    }, inplace=True)
    return df


def add_paper_reference_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Append paper's Table I as reference rows."""
    paper_rows = []
    for model, vals in PAPER_REFERENCE.items():
        row = {
            "Model":         f"{model} [PAPER]",
            "Accuracy":      vals["accuracy"],
            "Semantic Dist": vals["semantic_distance"],
            "Norm Sem Dist": vals["normalized_semantic_distance"],
        }
        paper_rows.append(row)
    return pd.concat([df, pd.DataFrame(paper_rows)], ignore_index=True)


def save_results_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save results DataFrame to tables/ directory."""
    out_path = config.TABLES_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"[evaluation] Results saved to {out_path}")
    return out_path


def print_summary_table(df: pd.DataFrame) -> None:
    """Pretty-print results to stdout."""
    cols = ["Model", "Accuracy", "Acc Std", "F1", "Semantic Dist", "Norm Sem Dist"]
    available = [c for c in cols if c in df.columns]
    print("\n" + "=" * 90)
    print(df[available].to_string(index=False))
    print("=" * 90)
