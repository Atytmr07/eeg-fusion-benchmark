"""Evaluation metrics shared by the Bonn and CHB-MIT pipelines."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def metrics(y_true: np.ndarray, prob: np.ndarray, ncls: int) -> dict:
    """Classification metrics from predicted class probabilities (N, ncls)."""
    pred = prob.argmax(1)
    out = {
        "acc": accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
    }
    try:
        if ncls == 2:
            out["auc"] = roc_auc_score(y_true, prob[:, 1])
        else:
            out["auc"] = roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
    except ValueError:
        out["auc"] = np.nan

    # Proper scoring rules. They are threshold-free and continuous, so they vary less
    # across folds than macro F1 and can resolve smaller differences.
    prob_c = np.clip(prob, 1e-7, 1.0)
    prob_c = prob_c / prob_c.sum(axis=1, keepdims=True)
    out["log_loss"] = log_loss(y_true, prob_c, labels=list(range(ncls)))
    onehot = np.eye(ncls)[y_true]
    out["brier"] = float(((prob - onehot) ** 2).sum(axis=1).mean())

    per_f1 = f1_score(y_true, pred, average=None, labels=list(range(ncls)), zero_division=0)
    per_rec = recall_score(y_true, pred, average=None, labels=list(range(ncls)), zero_division=0)
    for c in range(ncls):
        out[f"f1_c{c}"] = per_f1[c]
        out[f"recall_c{c}"] = per_rec[c]

    # The ictal class is always the last index.
    out["recall_ictal"] = per_rec[ncls - 1]
    out["cm"] = confusion_matrix(y_true, pred, labels=list(range(ncls))).tolist()
    return out


def clinical_metrics(y_true: np.ndarray, prob: np.ndarray, hours: float) -> dict:
    """Sensitivity, specificity and false alarms for binary seizure detection.

    WARNING: false alarms per hour are computed on the subsampled time base (4
    non-ictal windows kept per ictal window), so they are NOT the clinical rate. They
    are only for comparing models on the same windows.
    """
    pred = prob.argmax(1)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "false_alarms": fp,
        "fa_per_hour_subsampled": fp / hours if hours > 0 else float("nan"),
    }
