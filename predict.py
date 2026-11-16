#!/usr/bin/env python
"""predict.py — entregável do PA3.

Uso:
    python predict.py -imagens PASTA -saida preds.csv

Escreve um CSV com colunas `image_id,label` (label = nome da classe de
`data/labels/classes.csv`). Roda em CPU, sem internet, e imprime o número
de parâmetros do modelo.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import get_class_names
from src.data.transforms import supervised_eval
from src.models import build_model, count_parameters


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entregável: predição em CPU")
    p.add_argument("-imagens", "--imagens", required=True,
                   help="Pasta com as imagens a classificar")
    p.add_argument("-saida", "--saida", required=True,
                   help="Caminho do CSV de saída")
    p.add_argument("--pesos", default=None,
                   help="Checkpoint .pt (default: <run_dir>/best.pt)")
    p.add_argument("--run_dir", default="runs/student_perclass_resnet18_b5_s2",
                   help="Run dir do modelo entregue (default: melhor seed)")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tta", action="store_true",
                   help="Test-time augmentation (flip horizontal)")
    p.add_argument("--device", default="cpu",
                   help="cpu (padrão) ou cuda")
    return p.parse_args()


def resolve_weights(args) -> Path:
    if args.pesos:
        p = Path(args.pesos)
        if not p.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {p}")
        return p
    default = Path(args.run_dir) / "best.pt"
    if default.exists():
        return default
    raise FileNotFoundError(
        f"Esperava {default}. Use --pesos para apontar outro checkpoint."
    )


def load_image_ids(folder: Path) -> list[tuple[str, Path]]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Não é uma pasta: {folder}")
    pairs = sorted(
        (p.stem, p) for p in folder.iterdir() if p.suffix in IMAGE_EXTS
    )
    if not pairs:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {folder}")
    return pairs


@torch.no_grad()
def predict_batch(model, images, device, img_size, tta=False):
    tf = supervised_eval(img_size=img_size)
    tensors = torch.stack([tf(im) for im in images]).to(device)
    logits = model(tensors)
    probs = F.softmax(logits, dim=1)
    if tta:
        logits_flip = model(torch.flip(tensors, dims=[3]))
        probs = 0.5 * (probs + F.softmax(logits_flip, dim=1))
    return probs.cpu()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    class_names = get_class_names()
    num_classes = len(class_names)

    weights = resolve_weights(args)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = build_model(args.backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    n_params = count_parameters(model, trainable_only=False)
    print(f"[model] {args.backbone}  params={n_params:,} (~{n_params/1e6:.2f}M)")
    print(f"[model] pesos={weights}  device={device}  tta={args.tta}")

    folder = Path(args.imagens)
    image_ids = load_image_ids(folder)
    print(f"[data] {len(image_ids)} imagens em {folder}")

    rows = []
    t0 = time.time()
    for i in range(0, len(image_ids), args.batch_size):
        batch = image_ids[i : i + args.batch_size]
        imgs = [Image.open(p).convert("RGB") for _, p in batch]
        probs = predict_batch(model, imgs, device, args.img_size, tta=args.tta)
        preds = probs.argmax(dim=1).tolist()
        for (iid, _), pred in zip(batch, preds):
            rows.append({
                "image_id": iid,
                "label": class_names[int(pred)],
            })
        for im in imgs:
            im.close()
        done = min(i + args.batch_size, len(image_ids))
        if done % 50 == 0 or done == len(image_ids):
            print(f"  {done}/{len(image_ids)}  ({time.time() - t0:.1f}s)")

    out = Path(args.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[done] {len(rows)} predições em {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()