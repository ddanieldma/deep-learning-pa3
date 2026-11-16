"""Construção de prompts para o professor zero-shot (Trilha C).

Este módulo é a "cola" entre o CSV de classes do ENA24 e o foundation model.
Ele:
  * carrega os nomes das 21 classes;
  * humaniza snake_case -> linguagem natural;
  * adiciona sinônimos e nomes científicos (útil para BioCLIP);
  * monta um ensemble de templates por classe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Mapeamentos específicos do ENA24
# ---------------------------------------------------------------------------
# Nomes científicos para BioCLIP/BioCLIP 2 (treinado em taxonomia).
SCIENTIFIC_NAMES: dict[str, str] = {
    "american_black_bear": "Ursus americanus",
    "american_crow": "Corvus brachyrhynchos",
    "bird": "Aves",
    "bobcat": "Lynx rufus",
    "chicken": "Gallus gallus domesticus",
    "coyote": "Canis latrans",
    "dog": "Canis lupus familiaris",
    "domestic_cat": "Felis catus",
    "eastern_chipmunk": "Tamias striatus",
    "eastern_cottontail": "Sylvilagus floridanus",
    "eastern_fox_squirrel": "Sciurus niger",
    "eastern_gray_squirrel": "Sciurus carolinensis",
    "grey_fox": "Urocyon cinereoargenteus",
    "horse": "Equus ferus caballus",
    "northern_raccoon": "Procyon lotor",
    "red_fox": "Vulpes vulpes",
    "striped_skunk": "Mephitis mephitis",
    "virginia_opossum": "Didelphis virginiana",
    "white_tailed_deer": "Odocoileus virginianus",
    "wild_turkey": "Meleagris gallopavo",
    "woodchuck": "Marmota monax",
}

# Sinônimos que ajudam CLIP/BioCLIP a ancorar a classe.
# Incluem variantes de ortografia (grey/gray) e nomes comuns alternativos.
SYNONYMS: dict[str, list[str]] = {
    "american_black_bear": ["black bear"],
    "american_crow": ["crow"],
    "bird": ["aves", "songbird"],
    "bobcat": ["red lynx", "bay lynx"],
    "chicken": ["domestic chicken", "hen", "rooster"],
    "coyote": ["prairie wolf"],
    "dog": ["domestic dog"],
    "domestic_cat": ["cat", "house cat"],
    "eastern_chipmunk": ["chipmunk"],
    "eastern_cottontail": ["cottontail rabbit"],
    "eastern_fox_squirrel": ["fox squirrel"],
    "eastern_gray_squirrel": ["gray squirrel", "grey squirrel"],
    "grey_fox": ["gray fox"],  # ortografia US
    "horse": ["equine", "domestic horse"],
    "northern_raccoon": ["raccoon", "common raccoon"],
    "red_fox": [],
    "striped_skunk": ["skunk"],
    "virginia_opossum": ["opossum", "possum"],
    "white_tailed_deer": ["deer"],
    "wild_turkey": ["turkey", "eastern wild turkey"],
    "woodchuck": ["groundhog", "wood chuck"],
}

DEFAULT_TEMPLATES: list[str] = [
    "a photo of a {}",
    "a camera trap photo of a {}",
    "a wildlife photo of a {}",
]

# Templates para nome científico: sem artigo, porque "a Ursus americanus" soa mal.
SCIENTIFIC_TEMPLATES: list[str] = [
    "a photo of {}",
    "a photo of the species {}",
]


# ---------------------------------------------------------------------------
# Leitura das classes
# ---------------------------------------------------------------------------
def load_class_names(classes_csv: str | Path | None = None) -> list[str]:
    """Lê `classes.csv` ou usa o loader padrão de `src.data`.

    Se `classes_csv` for `None`, delega para `src.data.get_class_names()`
    (mesma fonte do `train_supervised.py`). Se um caminho for passado,
    lê o CSV direto — útil para debug.
    """
    if classes_csv is None:
        from src.data import get_class_names
        return list(get_class_names())

    df = pd.read_csv(classes_csv)
    name_col = next(
        (c for c in ("label", "class", "name", "species", "class_name")
         if c in df.columns),
        df.columns[0],
    )
    names = df[name_col].astype(str).tolist()
    for idx_col in ("idx", "index", "id", "class_id"):
        if idx_col in df.columns:
            order = df[idx_col].astype(int).argsort()
            names = [names[i] for i in order]
            break
    return names


def humanize(name: str) -> str:
    """`american_black_bear` -> `american black bear`."""
    return name.replace("_", " ").strip().lower()


# ---------------------------------------------------------------------------
# Construção dos prompts
# ---------------------------------------------------------------------------
def build_class_prompts(
    class_names: Sequence[str],
    *,
    use_scientific: bool = False,
    use_synonyms: bool = True,
    templates: Sequence[str] | None = None,
    scientific_templates: Sequence[str] | None = None,
    ensemble: bool = True,
) -> list[list[str]]:
    """Devolve, para cada classe, uma lista de prompts.

    Se `ensemble=False`, devolve só o primeiro prompt de cada classe
    (útil para diagnóstico e para comparar com/sem ensemble).
    """
    templates = list(templates or DEFAULT_TEMPLATES)
    scientific_templates = list(scientific_templates or SCIENTIFIC_TEMPLATES)

    all_prompts: list[list[str]] = []

    for name in class_names:
        base_names = [humanize(name)]
        if use_synonyms:
            base_names += SYNONYMS.get(name, [])

        prompts: list[str] = []
        for base in base_names:
            for template in templates:
                prompts.append(template.format(base))

        if use_scientific and name in SCIENTIFIC_NAMES:
            sci = SCIENTIFIC_NAMES[name]
            for template in scientific_templates:
                prompts.append(template.format(sci))

        # Remove duplicatas preservando ordem.
        seen: set[str] = set()
        unique: list[str] = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        if not ensemble:
            unique = unique[:1]

        all_prompts.append(unique)

    return all_prompts