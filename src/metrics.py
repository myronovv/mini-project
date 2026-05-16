"""Метрики якості виявлення аномалій."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_ms_per_sample: float
    confusion: list[list[int]]

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "latency_ms_per_sample": round(self.latency_ms_per_sample, 4),
            "confusion_matrix": self.confusion,
        }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, latency_ms: float, n: int) -> EvaluationResult:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        latency_ms_per_sample=latency_ms / max(n, 1),
        confusion=cm,
    )
