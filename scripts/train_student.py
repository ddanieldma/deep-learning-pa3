#!/usr/bin/env python
"""Treina o aluno da Trilha C (destilação do foundation model).

Combina:
  * as 105 amostras rotuladas de ``rotulos_{budget}.csv`` (CE supervisionada);
  * as amostras do ``train`` com pseudo-rótulos do professor (CE dura/suave
    ou ponderada, conforme ``--noise``).

O ``val`` é usado normalmente para early stopping (é o aluno, não o professor).
O professor nunca vê o ``val`` — a matriz de transição do SELFIE é estimada
com ``test_confusion.csv`` e ``rotulos_{budget}.csv``.

Uso:
  # baseline (sem tratamento de ruído)
  python scripts/train_student.py --pseudo_path runs/teacher_bioclip2/train_pseudo.pt \\
      --teacher_dir runs/teacher_bioclip2 --noise none

  # confiança
  python scripts/train_student.py --pseudo_path runs/teacher_bioclip2/train_pseudo.pt \\
      --teacher_dir runs/teacher_bioclip2 --noise conf --tau 0.7

  # SELFIE
  python scripts/train_student.py --pseudo_path runs/teacher_bioclip2/train_pseudo.pt \\
      --teacher_dir runs/teacher_bioclip2 --noise selfie
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import (
    LabeledImageDataset,
    get_class_names,
    get_num_classes,
    labeled_dataframe,
    load_test_public,
    load_val,
)
from src.data.pseudo import PseudoLabelDataset
from src.data.transforms import supervised_eval, supervised_train
from src.models import build_model, count_parameters, get_param_groups
from src.train.metric import (
    AverageMeter,
    compute_metrics,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
)
from src.train.noise import estimate_transition_matrix


# ---------------------------------------------------------------------------
# Helpers (cópia enxuta do train_supervised.py)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    def f(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, criterion=None):
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib indisponível ({exc}); pulando.")
        return
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in (
        (axes[0], "loss", "Loss"),
        (axes[1], "macro_f1", "Macro F1"),
    ):
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] curvas em {path}")


def cycle(loader):
    """Iterador infinito sobre um DataLoader."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def pseudo_loss(mode: str, logits: torch.Tensor, batch):
    """Loss nos pseudo-rótulos, dependendo do modo do dataset."""
    if mode == "hard":
        _, y = batch
        return F.cross_entropy(logits, y.to(logits.device))
    if mode == "soft":
        _, probs = batch
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs.to(logits.device) * log_probs).sum(dim=-1).mean()
    if mode == "weighted":
        _, y, conf = batch
        y = y.to(logits.device)
        conf = conf.to(logits.device)
        ce = F.cross_entropy(logits, y, reduction="none")
        return (conf * ce).mean()
    raise ValueError(f"mode inválido: {mode}")


# ---------------------------------------------------------------------------
# SELFIE
# ---------------------------------------------------------------------------
def build_transition_matrix(args, budget_df, num_classes: int) -> np.ndarray:
    """Estima T[i,j] = P(pred=j | true=i) a partir de test + 105 rotulados."""
    test_cm_path = Path(args.teacher_dir) / "test_confusion.csv"
    if not test_cm_path.exists():
        raise FileNotFoundError(
            f"Esperava {test_cm_path}. Rode teacher_inference.py primeiro."
        )
    cm_test_df = __import__("pandas").read_csv(test_cm_path, index_col=0)
    cm_test = cm_test_df.values

    pseudo = torch.load(args.pseudo_path, map_location="cpu", weights_only=False)
    id_to_idx = {iid: i for i, iid in enumerate(pseudo["image_ids"])}
    idx = [id_to_idx[iid] for iid in budget_df["image_id"] if iid in id_to_idx]
    if len(idx) < len(budget_df):
        print(
            f"[selfie] atenção: {len(budget_df) - len(idx)} dos 105 rotulados "
            f"não estão no train_pseudo.pt"
        )
    probs_budget = pseudo["probs"][idx].numpy()
    pred_budget = probs_budget.argmax(axis=1)
    true_budget = budget_df["label"].values[: len(pred_budget)]

    cm_budget = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_budget, pred_budget):
        cm_budget[int(t), int(p)] += 1

    T = estimate_transition_matrix(
        cm_test,
        cm_budget,
        w_test=args.selfie_w_test,
        w_budget=args.selfie_w_budget,
        smoothing=args.smoothing,
    )
    np.savetxt(Path(args.out_dir) / "transition_matrix.csv", T, delimiter=",")
    print(f"[selfie] matriz T estimada; salva em {args.out_dir}/transition_matrix.csv")
    return T


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aluno da Trilha C (destilação)")
    # Dados
    p.add_argument("--pseudo_path", required=True,
                   help="runs/teacher_X/train_pseudo.pt")
    p.add_argument("--teacher_dir", required=True,
                   help="runs/teacher_X (para test_confusion.csv)")
    p.add_argument("--budget", type=int, choices=[1, 5, 20], default=5)
    # Modelo
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--no_pretrained", action="store_true")
    # Otimização
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pseudo_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int,
                   default=min(4, (os.cpu_count() or 2) // 2))
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr_head", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    # Pseudo-rótulos e ruído
    p.add_argument("--noise", default="none",
                   choices=["none", "conf", "margin", "perclass", "temp", "selfie"])
    p.add_argument("--tau", type=float, default=0.7,
                   help="limiar de confiança (usado com --noise conf)")
    p.add_argument("--margin", type=float, default=0.3,
                   help="margem top1-top2 (usado com --noise margin)")
    p.add_argument("--per_class_target", type=int, default=100,
                   help="top-K por classe (usado com --noise perclass)")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="temperatura do soft label (usado com --noise temp)")
    p.add_argument("--smoothing", type=float, default=1e-2,
                   help="suavização da matriz de transição (SELFIE)")
    p.add_argument("--selfie_w_test", type=float, default=1.0)
    p.add_argument("--selfie_w_budget", type=float, default=2.0,
                   help="peso dos 105 rotulados no SELFIE (>=1 p/ cobrir raras)")
    p.add_argument("--pseudo_weight", type=float, default=1.0,
                   help="lambda em loss = CE_sup + lambda * CE_pseudo")
    p.add_argument("--mode", default=None, choices=["hard", "soft", "weighted"],
                   help="default inferido de --noise")
    # Avaliação
    p.add_argument("--eval_split", default="test_public",
                   choices=["test_public", "val"])
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _preset_noise(args: argparse.Namespace) -> None:
    """Mapeia --noise para combinação de filtros/correção + mode."""
    if args.mode is None:
        args.mode = "soft" if args.noise in ("temp", "selfie") else "hard"

    # Cada preset ativa só os filtros relevantes.
    if args.noise == "none":
        args.tau = 0.0
        args.margin = 0.0
        args.per_class_target = 0
        args.temperature = 1.0
        args._use_selfie = False
    elif args.noise == "conf":
        args.margin = 0.0
        args.per_class_target = 0
        args.temperature = 1.0
        args._use_selfie = False
    elif args.noise == "margin":
        args.tau = 0.0
        args.per_class_target = 0
        args.temperature = 1.0
        args._use_selfie = False
    elif args.noise == "perclass":
        args.tau = 0.0
        args.margin = 0.0
        args.temperature = 1.0
        args._use_selfie = False
    elif args.noise == "temp":
        args.tau = 0.0
        args.margin = 0.0
        args.per_class_target = 0
        args._use_selfie = False
    elif args.noise == "selfie":
        args.tau = 0.0
        args.margin = 0.0
        args.per_class_target = 0
        args.temperature = 1.0
        args._use_selfie = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    _preset_noise(args)

    pretrained = not args.no_pretrained and args.backbone.lower() != "unet"
    if args.lr is None:
        args.lr = 1e-3 if args.backbone.lower() == "unet" else 3e-4
    if args.lr_head is None:
        args.lr_head = args.lr if args.backbone.lower() == "unet" else 1e-3

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(
        args.out_dir
        or f"runs/student_{args.noise}_{args.backbone}_b{args.budget}_s{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    set_seed(args.seed)
    print(f"[config] aluno={args.backbone}  noise={args.noise}  mode={args.mode}")
    print(f"[config] budget={args.budget}  seed={args.seed}  device={device}")
    print(f"[config] lambda_pseudo={args.pseudo_weight}  out_dir={out_dir}")

    num_classes = get_num_classes()
    class_names = get_class_names()

    # ---------------- Dados supervisionados ----------------
    train_tf = supervised_train(img_size=args.img_size)
    eval_tf = supervised_eval(img_size=args.img_size)

    sup_df = labeled_dataframe(f"rotulos_{args.budget}")
    val_df = load_val()
    test_df = load_test_public() if args.eval_split == "test_public" else val_df
    test_split = "test_public" if args.eval_split == "test_public" else "val"

    sup_ds = LabeledImageDataset(sup_df, split="train", transform=train_tf)
    val_ds = LabeledImageDataset(val_df, split="val", transform=eval_tf)
    test_ds = LabeledImageDataset(test_df, split=test_split, transform=eval_tf)

    # ---------------- Pseudo-rótulos ----------------
    T = None
    if args._use_selfie:
        T = build_transition_matrix(args, sup_df, num_classes)

    pseudo_ds = PseudoLabelDataset(
        pseudo_path=args.pseudo_path,
        split="train",
        transform=train_tf,
        tau=args.tau,
        margin=args.margin,
        per_class_target=args.per_class_target,
        temperature=args.temperature,
        transition_matrix=T,
        mode=args.mode,
        verbose=True,
    )

    print(f"[data] sup={len(sup_ds)}  pseudo={len(pseudo_ds)}  "
          f"val={len(val_ds)}  {test_split}={len(test_ds)}")

    if len(pseudo_ds) == 0:
        raise RuntimeError(
            "Nenhuma pseudo-amostra sobreviveu aos filtros. "
            "Baixe --tau/--margin/--per_class_target ou use --noise none."
        )

    sup_loader = DataLoader(
        sup_ds,
        batch_size=min(args.batch_size, len(sup_ds)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    pseudo_loader = DataLoader(
        pseudo_ds,
        batch_size=args.pseudo_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---------------- Modelo ----------------
    model = build_model(args.backbone, num_classes=num_classes,
                        pretrained=pretrained).to(device)
    n_params = count_parameters(model)
    print(f"[model] {args.backbone}  params={n_params:,} ({n_params/1e6:.2f}M)")

    param_groups = get_param_groups(model, lr=args.lr, lr_head=args.lr_head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = warmup_cosine(optimizer, args.warmup_epochs, args.epochs)
    criterion_sup = nn.CrossEntropyLoss()

    history = {
        "epoch": [], "lr": [],
        "train_loss": [], "train_sup_loss": [], "train_pseudo_loss": [],
        "train_macro_f1": [], "train_acc": [],
        "val_loss": [], "val_macro_f1": [], "val_acc": [],
    }
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    best_path = out_dir / "best.pt"

    # ---------------- Loop ----------------
    for epoch in range(args.epochs):
        model.train()
        meter_total = AverageMeter()
        meter_sup = AverageMeter()
        meter_psd = AverageMeter()
        y_true, y_pred = [], []

        sup_iter = cycle(sup_loader)

        for batch_psd in pseudo_loader:
            batch_sup = next(sup_iter)

            x_sup, y_sup = batch_sup
            x_sup = x_sup.to(device, non_blocking=True)
            y_sup = y_sup.to(device, non_blocking=True)

            x_psd = batch_psd[0].to(device, non_blocking=True)
            # Reempacota o resto do batch (labels ou probs/conf) intacto.
            batch_psd_dev = (x_psd, *batch_psd[1:])

            # Forward: empilha os dois batches num só para economizar overhead
            x_all = torch.cat([x_sup, x_psd], dim=0)
            logits_all = model(x_all)
            logits_sup = logits_all[: x_sup.size(0)]
            logits_psd = logits_all[x_sup.size(0):]

            loss_sup = criterion_sup(logits_sup, y_sup)
            loss_psd = pseudo_loss(args.mode, logits_psd, batch_psd_dev)
            loss = loss_sup + args.pseudo_weight * loss_psd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = x_sup.size(0) + x_psd.size(0)
            meter_total.update(loss.item(), n=n)
            meter_sup.update(loss_sup.item(), n=x_sup.size(0))
            meter_psd.update(loss_psd.item(), n=x_psd.size(0))

            preds = logits_sup.argmax(dim=1)
            y_true.extend(y_sup.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        train_m = compute_metrics(y_true, y_pred, num_classes)
        val_loss, val_f1, val_acc, _, _ = evaluate(
            model, val_loader, device, num_classes, criterion=criterion_sup
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss"].append(meter_total.avg)
        history["train_sup_loss"].append(meter_sup.avg)
        history["train_pseudo_loss"].append(meter_psd.avg)
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
                {"model": model.state_dict(), "epoch": epoch,
                 "val_macro_f1": val_f1},
                best_path,
            )
        else:
            epochs_since_improve += 1

        mark = " *" if improved else ""
        print(
            f"epoch {epoch:03d}  lr={current_lr:.2e}  "
            f"loss={meter_total.avg:.3f} "
            f"(sup={meter_sup.avg:.3f} psd={meter_psd.avg:.3f})  "
            f"val_loss={val_loss:.3f}  val_f1={val_f1:.3f}  "
            f"val_acc={val_acc:.3f}{mark}"
        )

        scheduler.step()

        if epochs_since_improve >= args.patience:
            print(f"[early-stop] sem melhora por {args.patience} épocas.")
            break

    # ---------------- Fim ----------------
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_curves(history, out_dir / "curves.png")

    if not best_path.exists():
        raise RuntimeError("best.pt não foi salvo — algo deu errado.")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"\n[final] melhor época={ckpt['epoch']}  "
          f"val_f1={ckpt['val_macro_f1']:.4f}")

    _, test_f1, test_acc, y_true, y_pred = evaluate(
        model, test_loader, device, num_classes, criterion=criterion_sup
    )
    print(f"[final] {test_split}: macro_f1={test_f1:.4f}  acc={test_acc:.4f}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"split": test_split, "macro_f1": test_f1,
                   "accuracy": test_acc}, f, indent=2)
    per_class_metrics(y_true, y_pred, class_names, num_classes).to_csv(
        out_dir / "test_per_class.csv", index=False
    )
    confusion_matrix(y_true, y_pred, num_classes, class_names).to_csv(
        out_dir / "test_confusion.csv"
    )

    config = vars(args).copy()
    config.pop("_use_selfie", None)
    config.update({
        "device": str(device),
        "pretrained": pretrained,
        "num_params": int(n_params),
        "n_supervised": len(sup_ds),
        "n_pseudo_after_filter": len(pseudo_ds),
        "best_epoch": int(ckpt["epoch"]),
        "best_val_macro_f1": float(ckpt["val_macro_f1"]),
        "final_test_macro_f1": float(test_f1),
        "final_test_accuracy": float(test_acc),
    })
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\n[done] tudo salvo em {out_dir}/")


if __name__ == "__main__":
    main()