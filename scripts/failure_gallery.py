#!/usr/bin/env python
"""Galeria de falhas e atalhos (Parte 3) + diagnóstico open-set.

Para um run de aluno (ex.: runs/student_perclass_resnet18_b5_s0) e o CSV
de rótulos do test_public:

  1. Infere no test_public, gera CAM por amostra.
  2. Seleciona 3 erros (noite / baixa confiança / confusão comum).
  3. Seleciona 5 acertos (centro / borda / difuso / aleatórios).
  4. Salva PNGs (original | CAM | overlay), um `report.md` com diagnóstico
     automático, e um `open_set_signal.csv` (max_prob, entropia, margem)
     que responde à pergunta da apresentação.

Uso:
  python scripts/failure_gallery.py \
    --run_dir runs/student_perclass_resnet18_b5_s0 \
    --teacher_dir runs/teacher_bioclip2 \
    --out_dir runs/student_perclass_resnet18_b5_s0/gallery
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import (
    get_class_names,
    get_image_path,
    load_test_public,
)
from src.data.transforms import supervised_eval
from src.interpret import GradCAM, cam_stats, denormalize, overlay_cam, resolve_target_layer
from src.models import build_model


# ---------------------------------------------------------------------------
# Modelo / inferência
# ---------------------------------------------------------------------------
def load_model(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(config["backbone"], num_classes=21, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


@torch.no_grad()
def infer_test_public(model, device, img_size, transform):
    df = load_test_public()
    ids = df["image_id"].astype(str).tolist()
    y_true = df["label"].astype(int).tolist()

    probs_list = []
    tensors = []
    for iid in ids:
        img = Image.open(get_image_path(iid, "test_public")).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        tensors.append(x.cpu())
        logits = model(x)
        probs_list.append(F.softmax(logits, dim=1)[0].cpu().numpy())
        img.close()

    probs = np.stack(probs_list)
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    return ids, np.array(y_true), preds, confs, probs, tensors


def brightness_of(image_id: str) -> float:
    img = np.asarray(Image.open(get_image_path(image_id, "test_public")).convert("L"),
                     dtype=np.float32)
    return float(img.mean())


# ---------------------------------------------------------------------------
# Salvar PNG
# ---------------------------------------------------------------------------
def save_triptych(
    path: Path,
    img_rgb: np.ndarray,
    cam: np.ndarray,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay = overlay_cam(img_rgb, cam, alpha=0.55)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("original (224×224)")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------
def select_failures(
    df: pd.DataFrame, top_teacher_pairs: set[tuple[int, int]]
) -> list[dict]:
    """3 erros: mais noturno, menor confiança, par de confusão mais comum."""
    errs = df[df["pred"] != df["true"]].copy().reset_index(drop=True)
    if errs.empty:
        return []
    errs["brightness"] = errs["image_id"].map(brightness_of)

    chosen: list[dict] = []
    picked_ids: set[str] = set()

    def _take(tag: str, row: pd.Series) -> None:
        chosen.append({"tag": tag, **row.to_dict()})
        picked_ids.add(row["image_id"])

    # 1) Noite
    r = errs.sort_values("brightness").iloc[0]
    _take("erro_noturno", r)

    # 2) Baixa confiança (entre os erros restantes)
    rest = errs[~errs["image_id"].isin(picked_ids)]
    if not rest.empty:
        r = rest.sort_values("conf").iloc[0]
        _take("erro_baixa_conf", r)

    # 3) Confusão comum
    rest = errs[~errs["image_id"].isin(picked_ids)]
    if not rest.empty:
        def _score(row):
            return 1 if (int(row["true"]), int(row["pred"])) in top_teacher_pairs else 0
        rest = rest.copy()
        rest["score"] = rest.apply(_score, axis=1)
        r = rest.sort_values(["score", "conf"], ascending=[False, False]).iloc[0]
        _take("erro_confusao", r)

    return chosen


def select_successes(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """5 acertos: centro, borda, difuso + 2 aleatórios de classes distintas."""
    ok = df[df["pred"] == df["true"]].copy().reset_index(drop=True)
    if ok.empty:
        return []

    # CoM e focused_area precisam do CAM -> fazemos isso depois, mas o usuário
    # pediu CAM para todos. Selecionamos por heurística fraca e depois
    # classificamos em "centro/borda/difuso" com base no CAM computado.
    ok = ok.sort_values("conf", ascending=False)
    chosen: list[dict] = []
    used_classes: set[int] = set()

    # Alvo inicial: 5 classes distintas
    for _, r in ok.iterrows():
        c = int(r["true"])
        if c in used_classes:
            continue
        chosen.append({"tag": "acerto_diverso", **r.to_dict()})
        used_classes.add(c)
        if len(chosen) >= n:
            break

    # Completa se não houver 5 classes
    if len(chosen) < n:
        for _, r in ok.iterrows():
            if r["image_id"] in {c["image_id"] for c in chosen}:
                continue
            chosen.append({"tag": "acerto_diverso", **r.to_dict()})
            if len(chosen) >= n:
                break
    return chosen


# ---------------------------------------------------------------------------
# Diagnóstico auto
# ---------------------------------------------------------------------------
def auto_diagnosis(row: dict, stats: dict, med_bright: float,
                   teacher_pairs: set[tuple[int, int]]) -> str:
    tag = row["tag"]
    b = row.get("brightness", 0.0)
    parts = []

    if tag == "erro_noturno":
        parts.append(f"Erro noturno (brightness={b:.0f}/255, mediana do test={med_bright:.0f}).")
    elif tag == "erro_baixa_conf":
        parts.append(f"Confiança baixa (max_prob={row['conf']:.2f}).")
    elif tag == "erro_confusao":
        parts.append(
            f"Confusão comum: {row['true_name']} → {row['pred_name']} "
            f"(par frequente na matriz do professor)."
        )

    if (int(row["true"]), int(row["pred"])) in teacher_pairs:
        parts.append("Esse par também é confundido pelo professor.")

    parts.append(
        f"CAM com CoM na célula {stats['grid_pos']} "
        f"(com_y={stats['com_y']:.0f}, com_x={stats['com_x']:.0f}); "
        f"focused_area={stats['focused_area']*100:.0f}%, "
        f"entropia={stats['entropy']:.2f}."
    )

    if stats["focused_area"] < 0.10:
        parts.append("Atenção muito focada (provavelmente animal pequeno ou artefato pontual).")
    elif stats["focused_area"] > 0.30:
        parts.append("Atenção difusa (o modelo não tem uma região dominante).")

    if stats["grid_pos"] in ("NW", "NE", "SW", "SE", "N", "S", "E", "W") and tag.startswith("acerto"):
        parts.append("CAM na periferia — possível atalho de fundo.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Open-set signal
# ---------------------------------------------------------------------------
def open_set_signal(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """max_prob, entropia e margem por imagem. Sinais de 'unknown'."""
    eps = 1e-12
    max_prob = probs.max(axis=1)
    logp = np.log(probs + eps)
    entropy = -(probs * logp).sum(axis=1) / np.log(probs.shape[1])
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return pd.DataFrame({
        "image_id": df["image_id"],
        "true_idx": df["true"],
        "pred_idx": df["pred"],
        "is_correct": (df["pred"] == df["true"]).astype(int),
        "max_prob": max_prob,
        "entropy": entropy,
        "margin": margin,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Galeria de falhas + open-set (Parte 3)")
    p.add_argument("--run_dir", required=True,
                   help="runs/student_<noise>_<backbone>_b<k>_s<s>")
    p.add_argument("--teacher_dir", default=None,
                   help="runs/teacher_X (opcional, para puxar pares de confusão)")
    p.add_argument("--out_dir", default=None,
                   help="default=<run_dir>/gallery")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--target_layer", default=None,
                   help="override do nome pontuado (ex.: layer4.1.conv2)")
    p.add_argument("--n_successes", type=int, default=5)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or run_dir / "gallery")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = out_dir / "failures"
    successes_dir = out_dir / "successes"
    failures_dir.mkdir(exist_ok=True)
    successes_dir.mkdir(exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_names = get_class_names()

    print(f"[config] run_dir={run_dir}  device={device}")
    model, config = load_model(run_dir)
    model.to(device)

    # Pares de confusão do professor (opcional)
    teacher_pairs: set[tuple[int, int]] = set()
    if args.teacher_dir:
        tp = Path(args.teacher_dir) / "analysis" / "top_confusion_pairs.csv"
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        if tp.exists():
            tdf = pd.read_csv(tp)
            for _, r in tdf.iterrows():
                teacher_pairs.add((name_to_idx[r["true"]], name_to_idx[r["pred"]]))
            print(f"[teacher] {len(teacher_pairs)} pares de confusão carregados")

    transform = supervised_eval(img_size=args.img_size)

    print("Inferência no test_public...")
    ids, y_true, preds, confs, probs, tensors = infer_test_public(
        model, device, args.img_size, transform
    )

    df = pd.DataFrame({
        "image_id": ids, "true": y_true, "pred": preds, "conf": confs,
    })
    df["true_name"] = df["true"].map(lambda i: class_names[int(i)])
    df["pred_name"] = df["pred"].map(lambda i: class_names[int(i)])

    # Open-set signal
    df_os = open_set_signal(df, probs)
    df_os.to_csv(out_dir / "open_set_signal.csv", index=False)
    _plot_open_set(df_os, out_dir / "open_set_signal.png")
    print(f"[ok] open_set_signal.csv  (p95 max_prob = {df_os['max_prob'].quantile(0.95):.3f}; "
          f"p05 max_prob = {df_os['max_prob'].quantile(0.05):.3f})")

    # Target layer
    target = resolve_target_layer(model, args.target_layer)
    print(f"[saliency] target layer = {type(target).__name__}")
    cam_maker = GradCAM(model, target)

    med_bright = float(np.median([brightness_of(i) for i in ids[:100]]))  # amostra p/ velocidade

    # ---------------- Falhas ----------------
    failures = select_failures(df, teacher_pairs)
    fail_report = []
    for i, row in enumerate(failures, 1):
        idx = ids.index(row["image_id"])
        x = tensors[idx].to(device)
        cam = cam_maker(x, class_idx=int(row["pred"]))
        stats = cam_stats(cam)
        img_rgb = denormalize(tensors[idx][0])
        title = (f"FALHA {i} ({row['tag']}) | true={row['true_name']} "
                 f"pred={row['pred_name']} conf={row['conf']:.2f}")
        save_triptych(failures_dir / f"failure_{i}.png", img_rgb, cam, title)
        diagnosis = auto_diagnosis(row, stats, med_bright, teacher_pairs)
        fail_report.append({"n": i, "image_id": row["image_id"], **row, **stats,
                            "diagnosis": diagnosis})
        print(f"[falha {i}] {row['image_id']}  true={row['true_name']} "
              f"pred={row['pred_name']} conf={row['conf']:.2f}")

    # ---------------- Acertos ----------------
    successes = select_successes(df, n=args.n_successes)
    ok_report = []
    for i, row in enumerate(successes, 1):
        idx = ids.index(row["image_id"])
        x = tensors[idx].to(device)
        cam = cam_maker(x, class_idx=int(row["pred"]))
        stats = cam_stats(cam)
        img_rgb = denormalize(tensors[idx][0])
        title = (f"ACERTO {i} | true=pred={row['true_name']} "
                 f"conf={row['conf']:.2f} | CoM={stats['grid_pos']}")
        save_triptych(successes_dir / f"success_{i}.png", img_rgb, cam, title)
        ok_report.append({"n": i, "image_id": row["image_id"], **row, **stats})
        print(f"[acerto {i}] {row['image_id']}  class={row['true_name']} "
              f"conf={row['conf']:.2f}  CoM={stats['grid_pos']}")

    cam_maker.close()

    # ---------------- Relatório ----------------
    _write_report(out_dir / "report.md",
                  run_dir, config, df, df_os, fail_report, ok_report)
    print(f"\n[done] galeria em {out_dir}/")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _write_report(path, run_dir, config, df, df_os, failures, successes):
    lines = []
    lines.append(f"# Galeria de falhas e atalhos — `{run_dir.name}`\n")
    lines.append(f"- backbone: `{config['backbone']}`")
    lines.append(f"- pseudo: `{config.get('pseudo_path', '?')}`")
    lines.append(f"- noise: `{config.get('noise', '?')}`  "
                 f"mode: `{config.get('mode', '?')}`")
    lines.append(f"- budget: {config.get('budget', '?')}  "
                 f"seed: {config.get('seed', '?')}")
    lines.append(f"- test_public: {config.get('final_test_macro_f1', '?')} F1 macro\n")

    lines.append("## Falhas (3)\n")
    for f in failures:
        lines.append(f"### Falha {f['n']} — `{f['image_id']}` ({f['tag']})\n")
        lines.append(f"![falha](failures/failure_{f['n']}.png)\n")
        lines.append(f"- **Verdadeira:** `{f['true_name']}`")
        lines.append(f"- **Prevista:** `{f['pred_name']}`  "
                     f"(confiança {f['conf']:.2f})")
        lines.append(f"- **Diagnóstico automático:** {f['diagnosis']}")
        lines.append("")

    lines.append("\n## Acertos (5)\n")
    for s in successes:
        hint = "provavelmente olhou para o animal" if s["grid_pos"] in ("C",) \
            and s["focused_area"] < 0.20 else \
            "possível atalho de fundo" if s["grid_pos"] in ("NW", "NE", "SW", "SE") else \
            "atenção difusa — inconclusivo visualmente"
        lines.append(f"### Acerto {s['n']} — `{s['image_id']}`\n")
        lines.append(f"![acerto](successes/success_{s['n']}.png)\n")
        lines.append(f"- **Classe (verdadeira = prevista):** `{s['true_name']}`  "
                     f"(confiança {s['conf']:.2f})")
        lines.append(f"- **CoM da CAM:** célula `{s['grid_pos']}`  "
                     f"(focused_area={s['focused_area']*100:.0f}%, "
                     f"entropia={s['entropy']:.2f})")
        lines.append(f"- **Hipótese:** {hint}.")
        lines.append("")

    lines.append("\n## Open-set (pergunta da apresentação)\n")
    lines.append(
        "Sinais de \"imagem fora das 21 classes\": `max_prob` baixo, `entropia` alta, "
        "`margin` baixo. Ver `open_set_signal.csv` e `open_set_signal.png`.\n"
    )
    lines.append(
        f"- p95 de `max_prob` no test_public: **{df_os['max_prob'].quantile(0.95):.3f}**\n"
        f"- p05 de `max_prob` no test_public: **{df_os['max_prob'].quantile(0.05):.3f}**\n"
        f"- p05 de `margin`: **{df_os['margin'].quantile(0.05):.3f}**\n"
    )
    lines.append(
        "**Sugestão para a apresentação:** uma imagem nova com `max_prob < p05` "
        "**e** `margin < p05` seria sinalizada como \"fora das 21 classes\". "
        "O aluno sozinho tem esse sinal fraco; um segundo sinal vem do "
        "**professor** (BioCLIP 2): se o teacher também estiver com confiança baixa "
        "**e** discordar do aluno, é duplo indício. Em produção: rejeitar a predição "
        "ou marcar como `unknown` para revisão humana.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_open_set(df_os, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in (
        (axes[0], "max_prob", "max softmax"),
        (axes[1], "entropy", "entropia"),
        (axes[2], "margin", "margem top1-top2"),
    ):
        ax.hist(df_os[col], bins=30, color="steelblue", alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Sinais de open-set no test_public")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()