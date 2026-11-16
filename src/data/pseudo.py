"""PseudoLabelDataset: imagens sem rótulo + probabilidades do professor.

Suporta:
  * `train_pseudo.pt` (gerado por `scripts/teacher_inference.py`)
  * `test_public_preds.csv` (diagnóstico do professor sobre o test_public)

Filtros (`tau`, `margin`, `per_class_target`) e correções
(`temperature`, `transition_matrix`) são aplicados **na construção**, então
`len(ds)` reflete o número real de pseudo-amostras usadas no treino.

Modos de saída em `__getitem__`:
  * ``"hard"``     -> ``(img, label_int)``
  * ``"soft"``     -> ``(img, probs_K)``
  * ``"weighted"`` -> ``(img, label_int, conf)``
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .reader import get_image_path
from ..train.noise import (
    confidence_filter,
    correct_labels,
    margin_filter,
    per_class_threshold,
    temperature_scale,
)


# ---------------------------------------------------------------------------
# Loader de pseudo-rótulos
# ---------------------------------------------------------------------------
def _load_pseudo(
    path: str | Path,
) -> tuple[list[str], torch.Tensor]:
    """Retorna ``(image_ids, probs)`` a partir de um `.pt` ou `.csv`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pseudo-rótulos não encontrados: {p}")

    if p.suffix == ".pt":
        data = torch.load(p, map_location="cpu", weights_only=False)
        image_ids = list(data["image_ids"])
        probs = data["probs"]
        if not isinstance(probs, torch.Tensor):
            probs = torch.as_tensor(probs, dtype=torch.float32)
        return image_ids, probs

    if p.suffix == ".csv":
        df = pd.read_csv(p)
        image_ids = df["image_id"].astype(str).tolist()
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        prob_cols = sorted(prob_cols, key=lambda x: int(x.split("_")[1]))
        probs = torch.tensor(df[prob_cols].values, dtype=torch.float32)
        return image_ids, probs

    raise ValueError(f"Formato de pseudo-rótulos desconhecido: {p.suffix}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PseudoLabelDataset(Dataset):
    """Dataset de imagens sem rótulo + pseudo-rótulos do professor.

    Parameters
    ----------
    pseudo_path : `.pt` (train) ou `.csv` (test_public_preds)
    split : ``'train'``, ``'val'`` ou ``'test_public'`` (pasta das imagens)
    transform : transform PIL -> tensor
    tau : limiar de confiança (>0 ativa o filtro)
    margin : margem top1-top2 (>0 ativa o filtro)
    per_class_target : se >0, mantém até K por classe predita (top-K por
        confiança). Combate o viés de frequência.
    temperature : >1 achata os probs (soft labels); 1 = sem efeito.
    transition_matrix : se fornecido (K,K), aplica correção SELFIE
        (``correct_labels``) antes da temperatura. Filtros rodam **antes**
        da correção, para refletir a confiança real do professor.
    mode : ``'hard'`` | ``'soft'`` | ``'weighted'``
    image_ids : se fornecido, restringe a estas imagens (ex.: as 105
        rotuladas, para diagnóstico do professor); caso contrário usa todas.
    """

    def __init__(
        self,
        pseudo_path: str | Path,
        split: str,
        transform=None,
        tau: float = 0.0,
        margin: float = 0.0,
        per_class_target: int = 0,
        temperature: float = 1.0,
        transition_matrix: Optional[np.ndarray] = None,
        mode: str = "hard",
        image_ids: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        raw_ids, raw_probs = _load_pseudo(pseudo_path)
        probs_np = raw_probs.numpy().astype(np.float64)

        # Restrição opcional a um subconjunto de image_ids.
        if image_ids is not None:
            id_to_idx = {iid: i for i, iid in enumerate(raw_ids)}
            keep_idx = [id_to_idx[i] for i in image_ids if i in id_to_idx]
            missing = [i for i in image_ids if i not in id_to_idx]
            if missing:
                raise ValueError(
                    f"{len(missing)} image_id(s) não estão no arquivo de "
                    f"pseudo-rótulos. Exemplo: {missing[:3]}"
                )
            raw_ids = [raw_ids[i] for i in keep_idx]
            probs_np = probs_np[keep_idx]

        # 1) Filtros (baseados nos probs crus do professor)
        mask = np.ones(len(raw_ids), dtype=bool)
        if tau > 0:
            mask &= confidence_filter(probs_np, tau)
        if margin > 0:
            mask &= margin_filter(probs_np, margin)
        if per_class_target > 0:
            mask &= per_class_threshold(probs_np, per_class_target)

        raw_ids = [iid for iid, keep in zip(raw_ids, mask) if keep]
        probs_np = probs_np[mask]
        raw_conf = probs_np.max(axis=1)

        # 2) Correção SELFIE
        if transition_matrix is not None:
            probs_np = correct_labels(probs_np, transition_matrix)

        # 3) Temperatura (só faz sentido em modos suaves)
        probs_np = temperature_scale(probs_np, temperature)

        self.image_ids = raw_ids
        self.probs = torch.tensor(probs_np, dtype=torch.float32)
        self.raw_conf = torch.tensor(raw_conf, dtype=torch.float32)
        self.split = split
        self.transform = transform
        self.mode = mode

        if verbose:
            print(
                f"[PseudoLabelDataset] {len(raw_ids)} amostras após filtro "
                f"(tau={tau}, margin={margin}, per_class={per_class_target}); "
                f"selfie={transition_matrix is not None}; "
                f"T={temperature}; mode={mode}"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        iid = self.image_ids[idx]
        img = Image.open(get_image_path(iid, self.split)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.mode == "hard":
            return img, int(self.probs[idx].argmax())
        if self.mode == "soft":
            return img, self.probs[idx]
        if self.mode == "weighted":
            return img, int(self.probs[idx].argmax()), float(self.raw_conf[idx])
        raise ValueError(f"mode inválido: {self.mode}")