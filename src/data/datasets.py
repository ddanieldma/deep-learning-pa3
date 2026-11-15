"""PyTorch Datasets para o PA3."""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .reader import get_image_path


class LabeledImageDataset(Dataset):
    """
    Dataset de imagens rotuladas.

    Devolve ``(img, label_int)``.

    Parameters
    ----------
    df : DataFrame com colunas ``image_id`` e ``label``.
    split : ``'train'``, ``'val'`` ou ``'test_public'``.
    transform : aplicado à imagem PIL (já em RGB).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])
        img = Image.open(get_image_path(image_id, self.split)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class UnlabeledImageDataset(Dataset):
    """
    Dataset de imagens sem rótulo. Devolve ``(img, image_id)``.

    Útil para inferência sobre ``test_private``/``test_public`` e para
    inspecionar as imagens do pool (train inteiro).
    """

    def __init__(
        self,
        image_ids: Sequence[str],
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.image_ids = list(image_ids)
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img = Image.open(get_image_path(image_id, self.split)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, image_id


class MultiViewImageDataset(Dataset):
    """
    Aplica cada transform de ``transforms`` à mesma imagem e devolve as views
    como tupla.

    - SimCLR: ``transforms=[aug_a, aug_b]`` -> ``(view_a, view_b)``
    - FixMatch: ``transforms=[weak, strong]`` -> ``(weak, strong)``
    - Se ``labels`` (dict ``image_id -> int``) for fornecido, o label entra
      como último elemento: ``(view_a, view_b, label)``.

    Útil também para avaliação TTA (test-time augmentation).
    """

    def __init__(
        self,
        image_ids: Sequence[str],
        split: str,
        transforms: Iterable[Callable],
        labels: Optional[dict[str, int]] = None,
    ):
        self.image_ids = list(image_ids)
        self.split = split
        self.transforms = list(transforms)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img = Image.open(get_image_path(image_id, self.split)).convert("RGB")
        views = tuple(t(img) for t in self.transforms)
        if self.labels is not None:
            return (*views, int(self.labels[image_id]))
        return views