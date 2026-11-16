#!/usr/bin/env python
"""Analisa onde o professor erra (obrigatório da Trilha C).

Lê `runs/teacher_X/test_public_preds.csv` + `test_confusion.csv` e produz:
  * `analysis/confusion_row_normalized.csv` — recall por classe (a "matriz de ruído")
  * `analysis/top_confusion_pairs.csv` — pares (true, pred) mais confundidos
  * `analysis/brightness_vs_error.csv` — proxy de "noite" por luminância
  * `analysis/per_class_teacher.csv` — P/R/F1 + suporte por classe
  * `analysis/brightness_vs_error.png` — scatter
  * (opcional com `--student_dir`) `analysis/student_vs_teacher_per_class.csv`
    e `analysis/student_vs_teacher_f1_delta.png`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from src.data import get_class_names, get_image_path
from src.train.metric import confusion_matrix, per_class_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Análise do professor")
    p.add_argument("--teacher_dir", required=True,
                   help="runs/teacher_X")
    p.add_argument("--test_preds", default=None,
                   help="default=<teacher_dir>/test_public_preds.csv")
    p.add_argument("--test_confusion", default=None,
                   help="default=<teacher_dir>/test_confusion.csv")
    p.add_argument("--student_dir", default=None,
                   help="opcional: runs/student_* (para comparar F1 por classe)")
    p.add_argument("--split", default="test_public")
    p.add_argument("--top_k_pairs", type=int, default=15)
    p.add_argument("--num_brightness_bins", type=int, default=5)
    return p.parse_args()


def load_brightness(image_ids: list[str], split: str) -> np.ndarray:
    """Luminância média (0-255) de cada imagem."""
    vals = []
    for iid in image_ids:
        try:
            img = Image.open(get_image_path(iid, split)).convert("L")
        except FileNotFoundError:
            vals.append(np.nan)
            continue
        vals.append(float(np.asarray(img, dtype=np.float32).mean()))
    return np.array(vals)


def main() -> None:
    args = parse_args()
    teacher_dir = Path(args.teacher_dir)
    analysis_dir = teacher_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    preds_path = Path(args.test_preds or teacher_dir / "test_public_preds.csv")
    conf_path = Path(args.test_confusion or teacher_dir / "test_confusion.csv")

    df = pd.read_csv(preds_path)
    class_names = get_class_names()
    K = len(class_names)

    # ---------------- Matriz de confusão normalizada por linha ----------------
    cm_raw = pd.read_csv(conf_path, index_col=0)
    cm_np = cm_raw.values
    row_sum = cm_np.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = pd.DataFrame(cm_np / row_sum,
                           index=class_names, columns=class_names)
    cm_norm.to_csv(analysis_dir / "confusion_row_normalized.csv")
    print(f"[analysis] matriz normalizada por linha salva")

    # ---------------- Top-K pares confundidos ----------------
    pairs = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j or cm_np[i, j] == 0:
                continue
            pairs.append({
                "true": true_name, "pred": pred_name,
                "count": int(cm_np[i, j]),
                "share_of_true": float(cm_np[i, j] / max(1, cm_np[i].sum())),
            })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    pd.DataFrame(pairs[: args.top_k_pairs]).to_csv(
        analysis_dir / "top_confusion_pairs.csv", index=False
    )
    print(f"[analysis] top-{args.top_k_pairs} pares confundidos salvos")

    # ---------------- Per-class teacher ----------------
    if "true_idx" in df.columns and "pred_idx" in df.columns:
        y_true = df["true_idx"].astype(int).tolist()
        y_pred = df["pred_idx"].astype(int).tolist()
        per_class_metrics(y_true, y_pred, class_names, K).to_csv(
            analysis_dir / "per_class_teacher.csv", index=False
        )
        print(f"[analysis] P/R/F1 por classe do professor salvos")

    # ---------------- Luminância vs erro ----------------
    image_ids = df["image_id"].astype(str).tolist()
    bright = load_brightness(image_ids, args.split)
    df_b = df.copy()
    df_b["brightness"] = bright
    df_b["is_correct"] = (df["pred_idx"] == df["true_idx"]).astype(int) \
        if "true_idx" in df.columns else np.nan

    # Média de luminância em acertos e erros
    summary = df_b.groupby("is_correct")["brightness"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary["is_correct"] = summary["is_correct"].map(
        {0: "erro", 1: "acerto"}
    )
    summary.to_csv(analysis_dir / "brightness_vs_error.csv", index=False)

    # Bins de luminância × taxa de acerto
    df_b = df_b.dropna(subset=["brightness"]).reset_index(drop=True)
    if len(df_b) > 0:
        df_b["bright_bin"] = pd.qcut(
            df_b["brightness"], q=args.num_brightness_bins, duplicates="drop"
        )
        bin_stats = df_b.groupby("bright_bin", observed=True).agg(
            n=("is_correct", "size"),
            acc=("is_correct", "mean"),
            mean_bright=("brightness", "mean"),
        ).reset_index()
        bin_stats.to_csv(analysis_dir / "accuracy_by_brightness_bin.csv",
                         index=False)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(df_b["brightness"], df_b["is_correct"],
                       alpha=0.15, s=8)
            ax.set_xlabel("luminância média (0–255)")
            ax.set_ylabel("acertou? (1=sim, 0=não)")
            ax.set_title("Professor: acerto vs. luminância")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(analysis_dir / "brightness_vs_error.png", dpi=120)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print(f"[plot] pulando scatter: {exc}")

    # ---------------- Comparação aluno vs professor ----------------
    if args.student_dir:
        student_dir = Path(args.student_dir)
        student_pc = pd.read_csv(student_dir / "test_per_class.csv")
        teacher_pc = pd.read_csv(analysis_dir / "per_class_teacher.csv")
        merged = teacher_pc.merge(
            student_pc, on="class_name", suffixes=("_teacher", "_student")
        )
        merged["f1_delta"] = merged["f1_student"] - merged["f1_teacher"]
        merged = merged.sort_values("f1_delta", ascending=False)
        merged.to_csv(analysis_dir / "student_vs_teacher_per_class.csv",
                      index=False)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["tab:green" if v >= 0 else "tab:red"
                      for v in merged["f1_delta"]]
            ax.bar(merged["class_name"], merged["f1_delta"], color=colors)
            ax.set_ylabel("ΔF1 (aluno − professor)")
            ax.set_title("Onde o aluno supera o professor")
            ax.axhline(0, color="black", linewidth=0.8)
            plt.xticks(rotation=60, ha="right")
            fig.tight_layout()
            fig.savefig(analysis_dir / "student_vs_teacher_f1_delta.png",
                        dpi=120)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print(f"[plot] pulando bar chart: {exc}")

    print(f"\n[done] análise em {analysis_dir}/")


if __name__ == "__main__":
    main()