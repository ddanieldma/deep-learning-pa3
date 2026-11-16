# scripts/plot_budget_curve.py
"""Gera o gráfico obrigatório da Parte 2 a partir de curva_orcamento.csv."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def agg(df, metodo):
    """Média ± std por budget."""
    g = df[df["metodo"] == metodo].groupby("budget")["macro_f1"]
    return g.mean(), g.std()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="curva_orcamento.csv")
    p.add_argument("--out", default="curva_orcamento.png")
    p.add_argument("--teacher_dir", default="runs/teacher_bioclip2")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["budget"] > 0]  # remove teto e professor (tratados à parte)

    fig, ax = plt.subplots(figsize=(8, 5))

    for metodo, color, label in [
        ("piso", "tab:gray", "Piso (sup. 105 rótulos)"),
        ("student_perclass", "tab:blue", "Aluno (destilação, perclass)"),
    ]:
        sub = df[df["metodo"] == metodo]
        if sub.empty:
            continue
        g = sub.groupby("budget")["macro_f1"]
        mean, std = g.mean(), g.std().fillna(0)
        xs = mean.index.values
        ax.errorbar(xs, mean.values, yerr=std.values, marker="o",
                    capsize=4, label=label, color=color, linewidth=2)

    # Linha do teto
    teto = df[df["metodo"] == "teto_all_labels"]["macro_f1"]
    if not teto.empty:
        ax.axhline(float(teto.iloc[0]), color="tab:red", linestyle="--",
                   label=f"Teto ({float(teto.iloc[0]):.3f})")

    # Linha do professor
    prof = df[df["metodo"] == "professor_zero_shot"]["macro_f1"]
    if not prof.empty:
        ax.axhline(float(prof.iloc[0]), color="tab:green", linestyle=":",
                   label=f"Professor zero-shot ({float(prof.iloc[0]):.3f})")

    ax.set_xscale("log")
    ax.set_xticks([1, 5, 20])
    ax.set_xticklabels(["1", "5", "20"])
    ax.set_xlabel("rótulos por classe")
    ax.set_ylabel("F1 macro (test_public)")
    ax.set_title("Curva de orçamento — Trilha C")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()