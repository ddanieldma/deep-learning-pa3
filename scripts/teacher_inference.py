#!/usr/bin/env python
"""Roda o professor da Trilha C e salva os artefatos obrigatórios.

Artefatos gerados em `--out`:
  * test_public_preds.csv   -> image_id, pred, pred_idx, conf, true, true_idx, prob_0..prob_K
  * test_metrics.json       -> macro_f1, accuracy, n
  * test_confusion.csv      -> matriz de confusão (linhas = verdade, colunas = pred)
  * test_per_class.csv      -> precision, recall, f1, support por classe
  * train_pseudo.pt         -> image_ids, probs (6505, 21), class_names, pred, conf
  * teacher_config.json     -> configuração usada

O professor NUNCA vê o `val`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Permite rodar como `python scripts/teacher_inference.py` sem instalar o pacote.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prompts import build_class_prompts, load_class_names  # noqa: E402
from src.models.teacher import TEACHER_REGISTRY, TeacherModel  # noqa: E402


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _norm_label(x) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def read_labels_csv(path: str | Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return {
        str(row.image_id): _norm_label(row.label)
        for row in df.itertuples(index=False)
    }


def resolve_image_path(root: Path, image_id: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = root / f"{image_id}{ext}"
        if p.exists():
            return p
    matches = list(root.glob(f"{image_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Imagem não encontrada para {image_id} em {root}")


def label_to_idx(label: str, name_to_idx: dict[str, int]) -> int:
    label = _norm_label(label)
    if label in name_to_idx:
        return name_to_idx[label]
    try:
        i = int(label)
        if 0 <= i < len(name_to_idx):
            return i
    except ValueError:
        pass
    raise ValueError(f"Rótulo desconhecido: {label}")


def infer_paths(
    teacher: TeacherModel,
    paths: list[Path],
    class_embs: torch.Tensor,
    batch_size: int,
    temperature: float,
    desc: str = "",
) -> torch.Tensor:
    all_probs: list[torch.Tensor] = []
    n = len(paths)

    for i in range(0, n, batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        probs = teacher.classify_images(
            images, class_embs, temperature=temperature
        )
        all_probs.append(probs.detach().cpu())

        for im in images:
            im.close()

        if (i // batch_size) % 10 == 0:
            print(f"  {desc}{min(i + batch_size, n)}/{n}")

    return torch.cat(all_probs, dim=0)


def compute_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]):
    labels = list(range(len(class_names)))
    return {
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferência do professor (Trilha C)")
    p.add_argument(
        "--teacher",
        default="bioclip2",
        choices=list(TEACHER_REGISTRY),
        help="Professor a usar",
    )
    p.add_argument("--model_id", default=None, help="Override do model_id (transformers)")
    p.add_argument("--pretrained", default=None, help="Override do pretrained (open_clip)")
    p.add_argument(
        "--open_clip_model",
        default=None,
        help="Override da arquitetura open_clip, ex.: ViT-B-16",
    )
    p.add_argument("--classes", default=None, help="CSV de classes. Se omitido, usa src.data.get_class_names().")
    p.add_argument("--test_csv", default="data/labels/test_public.csv")
    p.add_argument("--test_dir", default="data/images/test_public")
    p.add_argument("--train_dir", default="data/images/train")
    p.add_argument("--out", default="runs/teacher")
    p.add_argument(
        "--template",
        default=None,
        help="Template único. Se omitido, usa o ensemble padrão.",
    )
    p.add_argument("--no_ensemble", action="store_true")
    p.add_argument("--use_scientific", action="store_true")
    p.add_argument("--no_synonyms", action="store_true")
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="Limita o test_public")
    p.add_argument("--limit_train", type=int, default=None, help="Limita o train")
    p.add_argument("--no_train", action="store_true", help="Não gera train_pseudo.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.classes)
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    templates = [args.template] if args.template else None
    class_prompts = build_class_prompts(
        class_names,
        use_scientific=args.use_scientific,
        use_synonyms=not args.no_synonyms,
        templates=templates,
        ensemble=not args.no_ensemble,
    )

    print(f"Carregando professor {args.teacher}...")
    teacher = TeacherModel(
        args.teacher,
        device=args.device,
        model_id=args.model_id,
        pretrained=args.pretrained,
        open_clip_model=args.open_clip_model,
    )
    print(f"Professor em {teacher.device}")

    print("Codificando prompts de classe...")
    class_embs = teacher.class_embeddings(class_prompts)

    # ------------------------------------------------------------------
    # test_public
    # ------------------------------------------------------------------
    print("Lendo test_public...")
    labels = read_labels_csv(args.test_csv)
    items = list(labels.items())
    if args.limit is not None:
        items = items[: args.limit]

    test_ids = [k for k, _ in items]
    y_true = [label_to_idx(v, name_to_idx) for _, v in items]
    test_paths = [
        resolve_image_path(Path(args.test_dir), image_id) for image_id in test_ids
    ]

    print(f"Inferência no test_public ({len(test_paths)} imagens)...")
    probs = infer_paths(
        teacher,
        test_paths,
        class_embs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        desc="test ",
    )
    probs_np = probs.numpy()
    preds = probs_np.argmax(axis=1)
    confs = probs_np.max(axis=1)

    metrics = compute_metrics(y_true, preds.tolist(), class_names)
    print(f"F1 macro zero-shot: {metrics['macro_f1']:.4f}")
    print(f"Acurácia zero-shot: {metrics['accuracy']:.4f}")

    # preds
    df_pred = pd.DataFrame(
        {
            "image_id": test_ids,
            "pred": [class_names[i] for i in preds],
            "pred_idx": preds,
            "conf": confs,
            "true": [class_names[i] for i in y_true],
            "true_idx": y_true,
        }
    )
    for c, cname in enumerate(class_names):
        df_pred[f"prob_{c}"] = probs_np[:, c]
    df_pred.to_csv(out / "test_public_preds.csv", index=False)

    # métricas
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # confusão
    cm = confusion_matrix(
        y_true, preds, labels=list(range(len(class_names)))
    )
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out / "test_confusion.csv"
    )

    # por classe
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    pd.DataFrame(
        {
            "class": class_names,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
        }
    ).to_csv(out / "test_per_class.csv", index=False)

    # ------------------------------------------------------------------
    # train pseudo-rótulos
    # ------------------------------------------------------------------
    if not args.no_train:
        train_root = Path(args.train_dir)
        train_paths = sorted(
            p
            for p in train_root.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if args.limit_train is not None:
            train_paths = train_paths[: args.limit_train]

        train_ids = [p.stem for p in train_paths]
        print(f"Inferência no train ({len(train_paths)} imagens)...")
        train_probs = infer_paths(
            teacher,
            train_paths,
            class_embs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            desc="train ",
        )

        torch.save(
            {
                "image_ids": train_ids,
                "probs": train_probs,
                "class_names": class_names,
                "pred": train_probs.argmax(dim=-1),
                "conf": train_probs.max(dim=-1).values,
            },
            out / "train_pseudo.pt",
        )
        print(f"Salvo {out / 'train_pseudo.pt'}")

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    config = {
        "teacher": args.teacher,
        "class_names": class_names,
        "templates": class_prompts,
        "temperature": args.temperature,
        "use_scientific": args.use_scientific,
        "use_synonyms": not args.no_synonyms,
        "ensemble": not args.no_ensemble,
        "batch_size": args.batch_size,
    }
    with open(out / "teacher_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Artefatos salvos em {out}")


if __name__ == "__main__":
    main()