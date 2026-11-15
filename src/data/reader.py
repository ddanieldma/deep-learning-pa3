"""
Leitura dos dados do PA3.

Estrutura esperada:

    data/
      images/{train,val,test_public}/*.jpg
      labels/{classes,val,test_public,rotulos_1,rotulos_5,rotulos_20}.csv

Todos os CSVs de rótulos têm colunas ``image_id,label``. ``image_id`` é o
nome do arquivo sem ``.jpg``. ``classes.csv`` mapeia o índice inteiro
(``label``) para o nome da classe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_DIR


# ---------------------------------------------------------------------------
# Constantes de layout
# ---------------------------------------------------------------------------
IMAGES_DIR: Path = Path(DATA_DIR) / "images"
LABELS_DIR: Path = Path(DATA_DIR) / "labels"

SPLITS: tuple[str, ...] = ("train", "val", "test_public")
BUDGETS: tuple[int, ...] = (1, 5, 20)

# Nomes de arquivos de rótulos disponíveis em data/labels
LABEL_FILES: tuple[str, ...] = (
    "val",
    "test_public",
    "rotulos_1",
    "rotulos_5",
    "rotulos_20",
)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
def load_classes() -> pd.DataFrame:
    """
    Lê ``classes.csv`` e devolve DataFrame com colunas
    ``label`` (int) e ``class_name`` (str), ordenado por label.

    Tolera os formatos mais comuns:
      - ``label,class_name`` com ``label`` inteiro
      - ``class_name,label`` (ordem invertida)
      - índice do pandas + coluna de nomes (sem coluna de inteiro explícita)
    A coluna inteira é detectada por conteúdo, não por nome/posição.
    """
    df = pd.read_csv(LABELS_DIR / "classes.csv")
    cols = list(df.columns)

    # Detecta qual coluna é numérica e qual é a dos nomes
    int_col: str | None = None
    str_col: str | None = None
    for c in cols:
        try:
            pd.to_numeric(df[c], errors="raise")
            if int_col is None:
                int_col = c
        except (ValueError, TypeError):
            if str_col is None:
                str_col = c

    if int_col is not None and str_col is not None:
        out = df[[int_col, str_col]].copy()
        out.columns = ["label", "class_name"]
    elif len(cols) == 1:
        # Uma coluna só com os nomes; o índice do pandas vira o label.
        out = df.reset_index()
        out.columns = ["label", "class_name"]
    else:
        raise ValueError(f"Formato inesperado em classes.csv: {cols}")

    out["label"] = out["label"].astype(int)
    out["class_name"] = out["class_name"].astype(str)
    return out.sort_values("label").reset_index(drop=True)

def normalize_labels(
    df: pd.DataFrame,
    classes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Garante que a coluna ``label`` seja o índice inteiro da classe.

    Aceita CSVs em que ``label`` já é inteiro, é uma string numérica, ou é o
    nome da classe (como neste PA). No caso do nome, mapeia via ``classes.csv``
    (com fallback case-insensitive).
    """
    if classes_df is None:
        classes_df = load_classes()

    out = df.copy()

    # Já está inteiro?
    if pd.api.types.is_integer_dtype(out["label"]):
        return out

    # Tenta cast direto (label como string numérica, ex.: "5")
    try:
        out["label"] = out["label"].astype(int)
        return out
    except (ValueError, TypeError):
        pass

    # Mapeia nome -> índice
    name_to_label = dict(zip(classes_df["class_name"], classes_df["label"]))
    name_to_label_lower = {k.lower(): v for k, v in name_to_label.items()}

    mapped: list[int | None] = []
    for val in out["label"]:
        if val in name_to_label:
            mapped.append(name_to_label[val])
        elif isinstance(val, str) and val.lower() in name_to_label_lower:
            mapped.append(name_to_label_lower[val.lower()])
        else:
            mapped.append(None)
    out["label"] = mapped

    missing = out["label"].isna()
    if missing.any():
        unknown = df.loc[missing, "label"].unique().tolist()
        raise ValueError(
            f"Rótulos não encontrados em classes.csv: {unknown}. "
            f"Classes disponíveis: {classes_df['class_name'].tolist()}"
        )

    out["label"] = out["label"].astype(int)
    return out

def get_class_names() -> list[str]:
    """Nomes das classes na ordem do índice (0..20)."""
    return load_classes()["class_name"].tolist()


def get_num_classes() -> int:
    return len(load_classes())


def get_label_to_name() -> dict[int, str]:
    df = load_classes()
    return dict(zip(df["label"], df["class_name"]))


def get_name_to_label() -> dict[str, int]:
    df = load_classes()
    return dict(zip(df["class_name"], df["label"]))


# ---------------------------------------------------------------------------
# Rótulos
# ---------------------------------------------------------------------------
def load_label_csv(name: str) -> pd.DataFrame:
    """
    Lê um CSV de rótulos em ``data/labels`` e devolve DataFrame com
    ``image_id,label`` e ``label`` já como inteiro (índice da classe).

    ``name`` deve ser ``'val'``, ``'test_public'`` ou ``'rotulos_{1,5,20}'``.
    """
    path = LABELS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de rótulos não encontrado: {path}. "
            f"Esperado em {LABELS_DIR}."
        )
    return normalize_labels(pd.read_csv(path))


def load_budget(k: int) -> pd.DataFrame:
    """Carrega ``rotulos_{k}.csv`` para ``k`` em {1, 5, 20}."""
    if k not in BUDGETS:
        raise ValueError(f"k deve ser um de {BUDGETS}, recebi {k}.")
    return load_label_csv(f"rotulos_{k}")


def load_val() -> pd.DataFrame:
    return load_label_csv("val")


def load_test_public() -> pd.DataFrame:
    return load_label_csv("test_public")


def load_label_map(name: str) -> dict[str, int]:
    """``{image_id: label}`` para um CSV de rótulos."""
    df = load_label_csv(name)
    return dict(zip(df["image_id"], df["label"].astype(int)))


def split_for_label_csv(name: str) -> str:
    """Em qual pasta de imagens vive o CSV de rótulos ``name``."""
    if name in ("val", "test_public"):
        return name
    if name.startswith("rotulos_"):
        return "train"
    raise ValueError(f"Nome de CSV de rótulos desconhecido: {name}")


# ---------------------------------------------------------------------------
# Imagens
# ---------------------------------------------------------------------------
def get_image_path(image_id: str, split: str) -> Path:
    """Caminho completo da imagem ``image_id`` no split ``split``."""
    if split not in SPLITS:
        raise ValueError(f"split deve ser um de {SPLITS}, recebi {split!r}.")
    return IMAGES_DIR / split / f"{image_id}.jpg"


def list_image_ids(split: str) -> list[str]:
    """Lista os ``image_id`` (stems) das imagens de um split, ordenado."""
    if split not in SPLITS:
        raise ValueError(f"split deve ser um de {SPLITS}, recebi {split!r}.")
    folder = IMAGES_DIR / split
    if not folder.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {folder}")
    return sorted(p.stem for p in folder.glob("*.jpg"))


def labeled_dataframe(name: str) -> pd.DataFrame:
    """
    CSV de rótulos enriquecido com ``split`` e ``path`` (str).

    Útil para montar ``Dataset`` rapidamente:
    ``LabeledImageDataset(labeled_dataframe("rotulos_5"), split="train")``.
    """
    df = load_label_csv(name).copy()
    split = split_for_label_csv(name)
    df["split"] = split
    df["path"] = df["image_id"].map(
        lambda x: str(IMAGES_DIR / split / f"{x}.jpg")
    )
    return df


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def check_data_layout() -> None:
    """Verifica se pastas e CSVs esperados existem; levanta se faltar algo."""
    missing: list[str] = []
    for s in SPLITS:
        if not (IMAGES_DIR / s).is_dir():
            missing.append(str(IMAGES_DIR / s))
    for name in ("classes",) + LABEL_FILES:
        p = LABELS_DIR / f"{name}.csv"
        if not p.is_file():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Estrutura de dados incompleta. Faltando:\n  - "
            + "\n  - ".join(missing)
        )


if __name__ == "__main__":
    # Verificação rápida: rode `python -m src.data.reader` a partir da raiz.
    check_data_layout()
    print(f"Classes ({get_num_classes()}):", get_class_names())
    for s in SPLITS:
        print(f"{s}: {len(list_image_ids(s))} imagens")
    for k in BUDGETS:
        df = load_budget(k)
        print(
            f"rotulos_{k}: {len(df)} imagens rotuladas, "
            f"{df['label'].nunique()} classes"
        )