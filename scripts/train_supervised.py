#!/usr/bin/env python
"""
Treino supervisionado — Parte 0.

Uso:
  # 0(a) CNN pequena do zero (U-Net encoder), orçamento 5
  python scripts/train_supervised.py --backbone unet --budget 5

  # 0(b) ResNet-18 fine-tuning, orçamento 5, 3 seeds
  for s in 0 1 2; do
      python scripts/train_supervised.py --backbone resnet18 --budget 5 --seed $s
  done

  # 0(b) com MobileNetV3 e EfficientNet-B0
  python scripts/train_supervised.py --backbone mobilenetv3 --budget 5
  python scripts/train_supervised.py --backbone efficientnet_b0 --budget 5

  # Orçamentos diferentes (Parte 2 já usa isso)
  python scripts/train_supervised.py --backbone resnet18 --budget 1
  python scripts/train_supervised.py --backbone resnet18 --budget 20

Saídas em ``runs/<backbone>_b<budget>_s<seed>/``:
  - ``history.csv``            — losses e métricas por época (train/val)
  - ``curves.png``             — curvas de treino/validação
  - ``best.pt``                — checkpoint do melhor F1 macro no val
  - ``test_metrics.json``      — macro F1 e accuracy no ``test_public``
  - ``test_confusion.csv``     — matriz de confusão no ``test_public``
  - ``test_per_class.csv``     — precisão/recall/F1 por classe no ``test_public``
  - ``config.json``            — args efetivos + nº de parâmetros do modelo

Model selection acontece SÓ no ``val``. O ``test_public`` só é avaliado no
final, com o melhor checkpoint — regra do PA.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import (
    LabeledImageDataset,
    get_class_names,
    get_num_classes,
    labeled_dataframe,
    load_test_public,
    load_val,
)
from src.data.transforms import supervised_eval, supervised_train
from src.models import build_model, count_parameters, get_param_groups
from src.train.metric import (
    AverageMeter,
    compute_metrics,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Scheduler: aquecimento linear por ``warmup_epochs`` seguido de cosine
    até ``total_epochs``. ``lambda`` é multiplicado pelo LR base de cada grupo.
    """
    def f(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, criterion=None):
    """Devolve (loss_avg, macro_f1, accuracy, y_true, y_pred)."""
    model.eval()
    meter = AverageMeter()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            meter.update(criterion(logits, y).item(), n=x.size(0))
        preds = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    m = compute_metrics(y_true, y_pred, num_classes)
    return meter.avg, m["macro_f1"], m["accuracy"], y_true, y_pred


def plot_curves(history: dict, path: Path) -> None:
    """Salva `curves.png` com loss e macro F1 em treino/val."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib é opcional
        print(f"[plot] matplotlib indisponível ({exc}); pulando gráfico.")
        return

    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_macro_f1"], label="train")
    axes[1].plot(epochs, history["val_macro_f1"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro F1")
    axes[1].set_title("Macro F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] curvas em {path}")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino supervisionado — Parte 0 do PA3")
    p.add_argument("--backbone", default="resnet18",
                   help="unet | resnet18 | resnet34 | resnet50 | mobilenetv3 | efficientnet_b0")
    p.add_argument("--budget", type=int, choices=[1, 5, 20], default=5,
                   help="rótulos por classe (rotulos_{k}.csv)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=min(4, (os.cpu_count() or 2) // 2))
    p.add_argument("--lr", type=float, default=None,
                   help="LR do backbone; default=1e-3 (scratch) ou 3e-4 (pretrained)")
    p.add_argument("--lr_head", type=float, default=None,
                   help="LR da cabeça (só pretrained); default=1e-3 ou --lr")
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=15,
                   help="early stopping pelo macro F1 do val")
    p.add_argument("--no_pretrained", action="store_true",
                   help="ignora pesos ImageNet (útil para sanidade)")
    p.add_argument("--eval_split", default="test_public", choices=["test_public", "val"],
                   help="split de avaliação final (não use para seleção!)")
    p.add_argument("--eval_full_resize", action="store_true",
                   help="avaliação sem CenterCrop (Resize direto p/ 224x224)")
    p.add_argument("--out_dir", default=None,
                   help="default=runs/<backbone>_b<budget>_s<seed>")
    p.add_argument("--device", default=None,
                   help="cuda|cpu (default: cuda se disponível)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Defaults que dependem do backbone
    pretrained = not args.no_pretrained and args.backbone.lower() != "unet"
    if args.lr is None:
        args.lr = 1e-3 if args.backbone.lower() == "unet" else 3e-4
    if args.lr_head is None:
        args.lr_head = args.lr if args.backbone.lower() == "unet" else 1e-3

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    out_dir = Path(
        args.out_dir
        or f"runs/{args.backbone.lower()}_b{args.budget}_s{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    print(f"[config] backbone={args.backbone}  budget={args.budget}  seed={args.seed}")
    print(f"[config] device={device}  out_dir={out_dir}")
    print(f"[config] lr={args.lr}  lr_head={args.lr_head}  epochs={args.epochs}")

    num_classes = get_num_classes()
    class_names = get_class_names()

    # ---------------- Dados ----------------
    train_tf = supervised_train(img_size=args.img_size)
    eval_tf_ = (
        supervised_eval(img_size=args.img_size)
        if not args.eval_full_resize
        else __import__("src.data.transforms", fromlist=["eval_tf_full"]).eval_tf_full(args.img_size)
    )

    train_df = labeled_dataframe(f"rotulos_{args.budget}")
    val_df = load_val()
    test_df = load_test_public() if args.eval_split == "test_public" else val_df

    train_ds = LabeledImageDataset(train_df, split="train", transform=train_tf)
    val_ds = LabeledImageDataset(val_df, split="val", transform=eval_tf_)
    test_split = "test_public" if args.eval_split == "test_public" else "val"
    test_ds = LabeledImageDataset(test_df, split=test_split, transform=eval_tf_)

    print(
        f"[data] train={len(train_ds)}  val={len(val_ds)}  "
        f"{test_split}={len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---------------- Modelo ----------------
    model = build_model(args.backbone, num_classes=num_classes, pretrained=pretrained)
    model.to(device)
    n_params = count_parameters(model)
    print(f"[model] {args.backbone}  params={n_params:,} (~{n_params/1e6:.2f}M)")

    # ---------------- Otimização ----------------
    param_groups = get_param_groups(model, lr=args.lr, lr_head=args.lr_head)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay
    )
    scheduler = warmup_cosine(optimizer, args.warmup_epochs, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---------------- Histórico ----------------
    history: dict[str, list] = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_macro_f1": [],
        "train_acc": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_acc": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    best_path = out_dir / "best.pt"

    # ---------------- Loop ----------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = AverageMeter()
        y_true, y_pred = [], []

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=x.size(0))
            preds = logits.argmax(dim=1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

        train_m = compute_metrics(y_true, y_pred, num_classes)
        val_loss, val_f1, val_acc, _, _ = evaluate(
            model, val_loader, device, num_classes, criterion=criterion
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss"].append(train_loss.avg)
        history["train_macro_f1"].append(train_m["macro_f1"])
        history["train_acc"].append(train_m["accuracy"])
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        improved = val_f1 > best_val_f1 + 1e-4
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1},
                best_path,
            )
        else:
            epochs_since_improve += 1

        mark = " *" if improved else ""
        print(
            f"epoch {epoch:03d}  lr={current_lr:.2e}  "
            f"train_loss={train_loss.avg:.3f}  train_f1={train_m['macro_f1']:.3f}  "
            f"val_loss={val_loss:.3f}  val_f1={val_f1:.3f}  val_acc={val_acc:.3f}{mark}"
        )

        scheduler.step()

        if epochs_since_improve >= args.patience:
            print(f"[early-stop] sem melhora por {args.patience} épocas; parando.")
            break

    # ---------------- Salvar histórico + curvas ----------------
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_curves(history, out_dir / "curves.png")

    # ---------------- Avaliação final ----------------
    if not best_path.exists():
        raise RuntimeError("Checkpoint best.pt não foi salvo — algo deu errado.")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"\n[final] melhor época={ckpt['epoch']}  val_macro_f1={ckpt['val_macro_f1']:.4f}")

    _, test_f1, test_acc, y_true, y_pred = evaluate(
        model, test_loader, device, num_classes, criterion=criterion
    )
    print(f"[final] {test_split}: macro_f1={test_f1:.4f}  acc={test_acc:.4f}")

    metrics = {"split": test_split, "macro_f1": test_f1, "accuracy": test_acc}
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    per_class_metrics(y_true, y_pred, class_names, num_classes).to_csv(
        out_dir / "test_per_class.csv", index=False
    )
    confusion_matrix(y_true, y_pred, num_classes, class_names).to_csv(
        out_dir / "test_confusion.csv"
    )

    # ---------------- Config de reprodutibilidade ----------------
    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "pretrained": pretrained,
            "num_params": int(n_params),
            "best_epoch": int(ckpt["epoch"]),
            "best_val_macro_f1": float(ckpt["val_macro_f1"]),
            "final_test_macro_f1": float(test_f1),
            "final_test_accuracy": float(test_acc),
        }
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\n[done] tudo salvo em {out_dir}/")


if __name__ == "__main__":
    main()