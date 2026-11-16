# scripts/build_budget_curve.py
"""Agrega os resultados da Parte 2 em `curva_orcamento.csv`."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


RUNS = Path("runs")


def read_metrics(run_dir: Path) -> dict | None:
    m = run_dir / "test_metrics.json"
    c = run_dir / "config.json"
    if not m.exists():
        return None
    data = json.loads(m.read_text())
    cfg = json.loads(c.read_text()) if c.exists() else {}
    return {
        "macro_f1": float(data["macro_f1"]),
        "accuracy": float(data["accuracy"]),
        "epochs": int(cfg.get("best_epoch", -1)),
    }


PISO_RE = re.compile(r"^resnet18_b(\d+)_s(\d+)$")
STUDENT_RE = re.compile(r"^student_([a-z]+)_resnet18_b(\d+)_s(\d+)$")


def main() -> None:
    args = parse_args()
    rows: list[dict] = []

    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue

        m = PISO_RE.match(d.name)
        if m:
            metrics = read_metrics(d)
            if metrics is None:
                continue
            budget, seed = int(m.group(1)), int(m.group(2))
            rows.append({
                "metodo": "piso",
                "budget": budget,
                "seed": seed,
                **metrics,
            })
            continue

        m = STUDENT_RE.match(d.name)
        if m:
            noise, budget, seed = m.group(1), int(m.group(2)), int(m.group(3))
            if noise != args.method:
                continue
            metrics = read_metrics(d)
            if metrics is None:
                continue
            rows.append({
                "metodo": f"student_{noise}",
                "budget": budget,
                "seed": seed,
                **metrics,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Nenhum run encontrado em runs/.")

    # Linhas de referência: professor e teto
    teacher_metrics = read_metrics(Path(args.teacher_dir))
    refs = []
    if teacher_metrics is not None:
        refs.append({
            "metodo": "professor_zero_shot",
            "budget": 0,
            "seed": 0,
            **teacher_metrics,
        })
    refs.append({
        "metodo": "teto_all_labels",
        "budget": 10000,  # x grande p/ não empatar com o eixo
        "seed": 0,
        "macro_f1": 0.686,
        "accuracy": float("nan"),
        "epochs": -1,
    })
    df = pd.concat([df, pd.DataFrame(refs)], ignore_index=True)

    df = df.sort_values(["metodo", "budget", "seed"]).reset_index(drop=True)
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"[ok] {len(df)} linhas em {out}")
    print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="perclass    ",
                   help="tag `noise` do aluno a usar como método da Parte 2")
    p.add_argument("--teacher_dir", default="runs/teacher_bioclip2")
    p.add_argument("--out", default="curva_orcamento.csv")
    return p.parse_args()


if __name__ == "__main__":
    main()