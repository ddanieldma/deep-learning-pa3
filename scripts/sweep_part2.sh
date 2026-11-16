#!/usr/bin/env bash
# scripts/sweep_part2.sh
set -euo pipefail

PSEUDO="runs/teacher_bioclip2/train_pseudo.pt"
TEACHER="runs/teacher_bioclip2"
METHOD="perclass"          # <-- troque aqui pelo melhor tratamento

# --- Piso: budget 1 e 20 (budget 5 já existe em runs/resnet18_b5_s{0,1,2})
python scripts/train_supervised.py --backbone resnet18 --budget 1  --seed 0
python scripts/train_supervised.py --backbone resnet18 --budget 20 --seed 0

# --- Método: budget 5 com 3 seeds (a seed 0 pode ser reusada se você já rodou)
for s in 0 1 2; do
  python scripts/train_student.py --pseudo_path "$PSEUDO" --teacher_dir "$TEACHER" \
    --noise "$METHOD" --budget 5 --seed "$s"
done

# --- Método: budget 1 e 20, uma seed cada
python scripts/train_student.py --pseudo_path "$PSEUDO" --teacher_dir "$TEACHER" \
  --noise "$METHOD" --budget 1 --seed 0
python scripts/train_student.py --pseudo_path "$PSEUDO" --teacher_dir "$TEACHER" \
  --noise "$METHOD" --budget 20 --seed 0