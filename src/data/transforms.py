"""
Pipelines de augmentation do PA3.

Duas coisas importam muito neste PA:
  1. O animal frequentemente ocupa <2% do quadro — cortes agressivos (padrão
     do SimCLR) tendem a produzir pares positivos sem animal. Por isso a
     augmentation supervisionada é mais leve (``scale=(0.6, 1.0)``).
  2. Muitas fotos são noturnas/IR. ``ColorJitter`` com ``hue`` pequeno é
     o correto; saturação alta machuca menos essas imagens.

Exports:
  - ``supervised_train`` / ``supervised_eval`` — aliases ``train_tf`` / ``eval_tf``
  - ``eval_tf_full`` — variante sem CenterCrop (para testar o efeito do corte)
  - ``simclr_aug`` — view SimCLR (forte, configurável, usado na Trilha A)
  - ``simclr_aug_no_crop`` — variante do experimento obrigatório da Trilha A
  - ``fixmatch_weak`` / ``fixmatch_strong`` — views do FixMatch (Trilha B)
"""
from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _norm() -> T.Normalize:
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ---------------------------------------------------------------------------
# Supervisionado / fine-tuning
# ---------------------------------------------------------------------------
def supervised_train(
    img_size: int = 224,
    scale: tuple[float, float] = (0.6, 1.0),
    hflip: float = 0.5,
    jitter: float = 0.2,
) -> T.Compose:
    """Augmentation de treino — compatível com backbone ImageNet."""
    return T.Compose(
        [
            T.RandomResizedCrop(
                img_size, scale=scale, ratio=(3 / 4, 4 / 3), antialias=True
            ),
            T.RandomHorizontalFlip(p=hflip),
            T.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter, hue=0.05
            ),
            T.ToTensor(),
            _norm(),
        ]
    )


def supervised_eval(img_size: int = 224, resize: int = 256) -> T.Compose:
    """Avaliação padrão ImageNet (Resize 256 → CenterCrop 224)."""
    return T.Compose(
        [
            T.Resize(resize, antialias=True),
            T.CenterCrop(img_size),
            T.ToTensor(),
            _norm(),
        ]
    )


def eval_tf_full(img_size: int = 224) -> T.Compose:
    """
    Avaliação alternativa: redimensiona toda a imagem para 224x224 sem crop.

    Útil para checar se o CenterCrop está jogando fora o animal (problema
    real quando ele ocupa <2% do quadro e está fora do centro).
    """
    return T.Compose(
        [
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            _norm(),
        ]
    )


# Aliases curtos usados no restante do código
train_tf = supervised_train
eval_tf = supervised_eval


# ---------------------------------------------------------------------------
# Auto-supervisionado (SimCLR — Trilha A)
# ---------------------------------------------------------------------------
def simclr_aug(
    img_size: int = 224,
    scale: tuple[float, float] = (0.4, 1.0),
    jitter: float = 0.4,
    blur_p: float = 0.5,
    gray_p: float = 0.2,
) -> T.Compose:
    """View SimCLR com crop relativamente controlado (scale_min=0.4)."""
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=scale, antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply(
                [T.ColorJitter(0.4 * jitter, 0.4 * jitter, 0.4 * jitter, 0.1)],
                p=0.8,
            ),
            T.RandomGrayscale(p=gray_p),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=blur_p
            ),
            T.ToTensor(),
            _norm(),
        ]
    )


def simclr_aug_no_crop(img_size: int = 224, jitter: float = 0.4) -> T.Compose:
    """
    Experimento obrigatório da Trilha A: remove o RandomResizedCrop
    (aumentação que gera pares positivos sem animal). Mantém o resto.
    """
    return T.Compose(
        [
            T.Resize((img_size, img_size), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply(
                [T.ColorJitter(0.4 * jitter, 0.4 * jitter, 0.4 * jitter, 0.1)],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5
            ),
            T.ToTensor(),
            _norm(),
        ]
    )


# ---------------------------------------------------------------------------
# Semi-supervisionado (FixMatch — Trilha B)
# ---------------------------------------------------------------------------
def fixmatch_weak(img_size: int = 224, scale: tuple[float, float] = (0.5, 1.0)) -> T.Compose:
    """View fraca: gera o pseudo-rótulo."""
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=scale, antialias=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            _norm(),
        ]
    )


def fixmatch_strong(
    img_size: int = 224,
    scale: tuple[float, float] = (0.5, 1.0),
    n: int = 2,
    m: int = 10,
) -> T.Compose:
    """View forte (RandAugment + RandomErasing): alvo da consistência."""
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=scale, antialias=True),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=n, magnitude=m),
            T.ToTensor(),
            _norm(),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ]
    )