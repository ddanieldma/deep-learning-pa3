# AI_LOG

## Episódios
1. **Primeira carga do BioCLIP 2 falhou** — passava `model_name="ViT-B-16"` e
   `pretrained="hf-hub:..."` separados. Corrigimos consultando a doc do open_clip
   (o hf-hub deve ir como `model_name` único).
2. **`transformers` 5.x** — `get_text_features` retorna `BaseModelOutputWithPooling`,
   não tensor. Extração via `.pooler_output`.
3. **`classes.csv` do ENA24** — a lista de nomes científicos que havíamos montado
   estava baseada em classes erradas (ex.: `equine`, `opossum`, `skunk` não existem
   no dataset). Corrigimos cruzando com o CSV real.
4. **Trilha C: baseline `none` ficou em 0,40** (abaixo do professor 0,56).
   A IA sugeriu testar `perclass` primeiro, o que deu 0,55. Escolhemos como método.

## Como usamos
- Geração inicial de `src/data/prompts.py`, `src/models/teacher.py`,
  `src/data/pseudo.py`, `src/train/noise.py`, `scripts/train_student.py`,
  `scripts/analyze_teacher.py`, `scripts/failure_gallery.py`, `predict.py`.
- Revisão dos erros e ajustes (acima).
- Toda a engenharia de dados (`reader.py`, `transforms.py`, `datasets.py`,
  `train_supervised.py`, `build_budget_curve.py`) foi feita antes/manualmente
  e não foi reescrita.