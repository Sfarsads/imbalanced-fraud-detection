"""Shared evaluation for EEE 405 Team 17. Everyone imports evaluate(); don't rewrite it."""
import json
import os

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_prob, threshold=0.5):
    """Fraud-class (label 1) metrics on the test set.

    y_prob: predicted probability of the positive (fraud) class, shape (n,).
    threshold: decision cutoff for precision/recall/F1/confusion matrix
               (default 0.5; Person 3 passes the tuned one).
    PR-AUC (average_precision_score) uses the raw scores, so it ignores the threshold.

    Returns a JSON-serializable dict with keys:
    precision, recall, f1, pr_auc, confusion_matrix ([[TN, FP], [FN, TP]]).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def save_results(model_name, metrics, results_dir="results", y_prob=None):
    """Write results/<model_name>.json with keys: model, precision, recall, f1,
    pr_auc, confusion_matrix. If y_prob is given, also saves
    results/<model_name>_probs.npy so the PR-curve plot can be drawn later."""
    os.makedirs(results_dir, exist_ok=True)
    out = {"model": model_name, **metrics}
    with open(os.path.join(results_dir, f"{model_name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    if y_prob is not None:
        np.save(os.path.join(results_dir, f"{model_name}_probs.npy"), np.asarray(y_prob))
    return out
