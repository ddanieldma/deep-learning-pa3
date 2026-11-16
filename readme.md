# PA3 — Trilha C (professor BioCLIP 2 → aluno ResNet-18)

## Ambiente
Python 3.10+; instalar com `uv sync` ou `pip install -r requirements.txt`.
Para o professor: `pip install open_clip_torch transformers`.

## Comando que treina
python scripts/train_student.py \
  --pseudo_path runs/teacher_bioclip2/train_pseudo.pt \
  --teacher_dir runs/teacher_bioclip2 \
  --noise perclass --per_class_target 100 --seed 2

## Comando que avalia (entregável)
python predict.py -imagens data/images/test_public -saida preds.csv

## Estrutura
- `runs/teacher_bioclip2/` — inferência do professor + análise
- `runs/student_perclass_resnet18_b5_s2/` — modelo entregue
- `curva_orcamento.csv` — Parte 2
- `runs/student_perclass_resnet18_b5_s2/gallery/` — Parte 3