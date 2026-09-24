"""Metrikler ve tek bir (tekrar, fold) için tüm modellerin değerlendirilmesi."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def metrics(y_true: np.ndarray, prob: np.ndarray, ncls: int) -> dict:
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

    # Uygun skorlama kuralları: eşiksiz ve sürekli oldukları için macro F1'e göre
    # fold'lar arası varyansları daha düşüktür (Faz 0, madde 0.5).
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

    # En kritik sınıf (son indeks = iktal) için ayrıca duyarlılık
    out["recall_ictal"] = per_rec[ncls - 1]
    out["cm"] = confusion_matrix(y_true, pred, labels=list(range(ncls))).tolist()
    return out
