"""
Métricas de classificação para o PA3.

O F1 macro é a métrica principal do PA (pesa cavalo igual a urso) e é o
critério de seleção de modelo. Acurácia também é reportada. Todos os
helpers devolvem tipos simples / DataFrames para facilitar salvar em CSV.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix as _sk_confusion_matrix,
)
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Escalares
# ---------------------------------------------------------------------------
def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int | None = None,
) -> float:
    """
    F1 macro. ``num_classes`` garante que classes ausentes no batch não
    distorcem a média.
    """
    labels = list(range(num_classes)) if num_classes is not None else None
    return float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int | None = None,
) -> dict[str, float]:
    return {
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "accuracy": accuracy(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------
def per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    num_classes: int | None = None,
) -> pd.DataFrame:
    """
    Precisão, recall, F1 e suporte por classe.

    Vai ser útil na Parte 2 ("quebrem o ganho por classe") e na Parte 3
    ("onde o professor erra de forma sistemática").
    """
    k = num_classes if num_classes is not None else len(class_names)
    labels = list(range(k))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "label": labels,
            "class_name": list(class_names)[:k],
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": sup,
        }
    )


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    class_names: Sequence[str] | None = None,
    normalize: str | None = None,  # None | "true" | "pred" | "all"
) -> pd.DataFrame:
    """
    Matriz de confusão como DataFrame com nomes de classe em linhas e colunas.

    ``normalize="true"`` normaliza por linha (recall por classe).
    """
    labels = list(range(num_classes))
    cm = _sk_confusion_matrix(
        y_true, y_pred, labels=labels, normalize=normalize
    )
    names = (
        list(class_names)[:num_classes]
        if class_names is not None
        else [str(i) for i in labels]
    )
    return pd.DataFrame(cm, index=names, columns=names)


# ---------------------------------------------------------------------------
# Utilidades de treino
# ---------------------------------------------------------------------------
class AverageMeter:
    """Média móvel para perda/accuracy ao longo da época."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0