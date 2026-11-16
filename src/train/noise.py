"""Tratamento do ruído nos pseudo-rótulos do professor (Trilha C).

Funções puras, sem estado: recebem `np.ndarray`/`torch.Tensor` e devolvem o
mesmo tipo, para permitir ablação sem tocar no treino.

Notação:
  probs : (N, K) probabilidades do professor (após softmax)
  T     : (K, K) matriz de transição; T[i, j] = P(prof prediz j | verdade é i)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
def confidence_filter(probs: np.ndarray, tau: float) -> np.ndarray:
    """Máscara booleana: True onde ``max_c probs > tau``."""
    return probs.max(axis=1) > tau


def margin_filter(probs: np.ndarray, margin: float) -> np.ndarray:
    """Máscara booleana: True onde ``p(1) - p(2) > margin``."""
    if probs.shape[1] < 2:
        return np.ones(len(probs), dtype=bool)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]) > margin


def per_class_threshold(
    probs: np.ndarray,
    target_per_class: int,
    min_tau: float = 0.0,
) -> np.ndarray:
    """Seleciona até ``target_per_class`` amostras por classe predita.

    Escolhe as de maior confiança; opcionalmente exige confiança mínima
    ``min_tau``. Combate o viés de frequência do professor (classes grandes
    dominam os pseudo-rótulos).
    """
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    mask = np.zeros(len(probs), dtype=bool)

    for c in range(probs.shape[1]):
        idx_c = np.where(preds == c)[0]
        if len(idx_c) == 0:
            continue
        eligible = idx_c[confs[idx_c] >= min_tau]
        if len(eligible) == 0:
            continue
        order = np.argsort(-confs[eligible])
        keep = eligible[order[:target_per_class]]
        mask[keep] = True

    return mask


# ---------------------------------------------------------------------------
# Soft labels e temperatura
# ---------------------------------------------------------------------------
def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Aquece (>1 achata) ou esfria (<1 aguça) um softmax já calculado.

    Reaplica softmax sobre ``log(probs) / T`` para preservar a normalização.
    """
    if temperature == 1.0:
        return probs
    logits = np.log(np.clip(probs, 1e-12, None)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# SELFIE simplificado: matriz de transição + correção
# ---------------------------------------------------------------------------
def estimate_transition_matrix(
    cm_test: np.ndarray,
    cm_budget: np.ndarray | None = None,
    w_test: float = 1.0,
    w_budget: float = 1.0,
    smoothing: float = 1e-2,
) -> np.ndarray:
    """Estima ``T[i, j] = P(pred=j | true=i)``.

    Combina (opcionalmente) a confusão do professor no ``test_public``
    (grande, mas cavalo tem só 4 amostras) com a confusão nos 105
    rotulados (pequena, mas cobre todas as 21 classes). ``smoothing``
    evita linhas com zeros — crítico para a inversão ficar estável.
    """
    cm = w_test * np.asarray(cm_test, dtype=np.float64)
    if cm_budget is not None:
        cm = cm + w_budget * np.asarray(cm_budget, dtype=np.float64)
    cm = cm + smoothing
    return cm / cm.sum(axis=1, keepdims=True)


def correct_labels(
    probs: np.ndarray,
    transition_matrix: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Desfaz o ruído dos pseudo-rótulos via SELFIE simplificado.

    Seja ``T[i, j] = P(pred=j | true=i)`` e ``q_j = P(pred=j | x)`` (probs do
    professor). Então ``q = T^T p``, onde ``p_i = P(true=i | x)``. Resolvemos
    ``T^T p = q`` para cada linha, clipamos em [0, ∞) e renormalizamos.

    A inversa explícita é evitada; ``np.linalg.solve`` é mais estável.
    Retorna (N, K) em float64 e já normalizado.
    """
    probs = np.asarray(probs, dtype=np.float64)
    T = np.asarray(transition_matrix, dtype=np.float64)
    K = T.shape[0]
    T_reg = T + eps * np.eye(K)

    # p = (T^T)^{-1} q  ->  resolve T^T p = q para cada coluna de probs^T
    corrected = np.linalg.solve(T_reg.T, probs.T).T  # (N, K)

    corrected = np.clip(corrected, 0.0, None)
    row_sum = corrected.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return corrected / row_sum