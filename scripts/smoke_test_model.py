"""
Smoke test do UNetEncoderClassifier.

Verifica:
  1. Contagem de parâmetros.
  2. Forward pass com tensor aleatório.
  3. Forward pass com imagens reais do conjunto rotulado (rotulos_5).
  4. Um passo de treino completo (forward + loss + backward + step).

Não treina nada de verdade — só garante que o modelo está plugado
corretamente e que os dados passam por ele.
"""
import sys
from pathlib import Path

# Garante que `src` seja importável quando rodamos `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchvision import transforms as T

from src.models import UNetEncoderClassifier


def main() -> None:
    print("=" * 60)
    print("SMOKE TEST — UNetEncoderClassifier")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Modelo + contagem de parâmetros
    # ------------------------------------------------------------------
    model = UNetEncoderClassifier(num_classes=21)
    n_params = model.num_parameters()
    print(f"\n[1] Parâmetros treináveis: {n_params:,} (~{n_params/1e6:.2f}M)")
    assert n_params < 11_000_000, "Modelo acima do orçamento tipo ResNet-18 (11M)."

    # Versão maior, opcional, só para mostrar que é configurável
    big = UNetEncoderClassifier(widths=(64, 128, 256, 512))
    print(f"    Variante widths=(64,128,256,512): {big.num_parameters():,} parâmetros")
    del big

    # ------------------------------------------------------------------
    # 2. Forward com tensor aleatório
    # ------------------------------------------------------------------
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"\n[2] Forward aleatório: {tuple(x.shape)} -> {tuple(y.shape)}")
    assert y.shape == (2, 21), f"Esperado (2, 21), obtive {tuple(y.shape)}"

    # Testa uma resolução diferente (imagens do PA têm proporções variadas;
    # o AdaptiveAvgPool2d lida com isso, mas os poolings exigem >= 32 px)
    x2 = torch.randn(1, 3, 320, 192)
    with torch.no_grad():
        y2 = model(x2)
    print(f"    Resolução 320x192: {tuple(x2.shape)} -> {tuple(y2.shape)}")
    assert y2.shape == (1, 21)

    # ------------------------------------------------------------------
    # 3. Forward com imagens reais
    # ------------------------------------------------------------------
    from src.data import LabeledImageDataset, labeled_dataframe

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = LabeledImageDataset(
        labeled_dataframe("rotulos_5"), split="train", transform=tf
    )
    n = min(4, len(ds))
    imgs = torch.stack([ds[i][0] for i in range(n)])
    labels = torch.tensor([ds[i][1] for i in range(n)])
    print(f"\n[3] Batch real: {tuple(imgs.shape)}, labels={labels.tolist()}, "
          f"min={imgs.min():.2f}, max={imgs.max():.2f}")

    model.eval()
    with torch.no_grad():
        logits = model(imgs)
        probs = logits.softmax(dim=1)
    print(f"    Logits: {tuple(logits.shape)}")
    print(f"    Soma de probs por linha: {probs.sum(dim=1).tolist()}")
    assert torch.allclose(probs.sum(dim=1), torch.ones(n), atol=1e-5)

    # ------------------------------------------------------------------
    # 4. Passo de treino
    # ------------------------------------------------------------------
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model(imgs), labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"\n[4] Passo de treino OK. Loss = {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSOU ✔")
    print("=" * 60)


if __name__ == "__main__":
    main()