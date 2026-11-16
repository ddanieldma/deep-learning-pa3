"""Métodos de saliência para a galeria de falhas (Parte 3).

Grad-CAM via hooks + oclusão como fallback. Funciona em qualquer CNN da
`build_model` (ResNet, MobileNetV3, EfficientNet, U-Net encoder).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Target layer
# ---------------------------------------------------------------------------
def find_last_conv(model: nn.Module) -> nn.Conv2d:
    """Última Conv2d do modelo (funciona em ResNet/MobileNet/EfficientNet)."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("Nenhuma Conv2d no modelo.")
    return last


def resolve_target_layer(
    model: nn.Module, override: Optional[str] = None
) -> nn.Module:
    """Aceita nome pontuado (``layer4.1.conv2``) ou auto-detecta."""
    if override is None:
        return find_last_conv(model)
    module = model
    for part in override.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM clássico (Selvaraju et al., 2017) via hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._h_fwd = target_layer.register_forward_hook(self._save_act)
        self._h_bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, gin, gout):
        self.gradients = gout[0].detach()

    def __call__(
        self, x: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)[0].cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        return cam

    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


# ---------------------------------------------------------------------------
# Oclusão (fallback, caso Grad-CAM falhe)
# ---------------------------------------------------------------------------
@torch.no_grad()
def occlusion_saliency(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    patch: int = 32,
    stride: int = 16,
    baseline: float = 0.0,
) -> np.ndarray:
    model.eval()
    _, _, H, W = x.shape
    base_prob = F.softmax(model(x), dim=1)[0, class_idx].item()

    sal = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch + 1, stride):
        for x0 in range(0, W - patch + 1, stride):
            x_occ = x.clone()
            x_occ[0, :, y:y + patch, x0:x0 + patch] = baseline
            prob = F.softmax(model(x_occ), dim=1)[0, class_idx].item()
            sal[y:y + patch, x0:x0 + patch] += max(0.0, base_prob - prob)
            count[y:y + patch, x0:x0 + patch] += 1

    mask = count > 0
    sal[mask] /= count[mask]
    if sal.max() > sal.min():
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal


# ---------------------------------------------------------------------------
# Estatísticas
# ---------------------------------------------------------------------------
def cam_stats(cam: np.ndarray) -> dict:
    H, W = cam.shape
    total = cam.sum()
    if total <= 0:
        return {
            "peak_y": H // 2, "peak_x": W // 2,
            "com_y": H / 2, "com_x": W / 2,
            "grid_pos": "C", "focused_area": 0.0, "entropy": 0.0,
        }
    ys, xs = np.mgrid[0:H, 0:W]
    peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)
    com_y = float((ys * cam).sum() / total)
    com_x = float((xs * cam).sum() / total)
    grid = _grid_3x3(com_y, com_x, H, W)
    focused_area = float((cam > 0.2 * cam.max()).mean())
    p = cam.flatten() / total
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum() / np.log(cam.size))
    return {
        "peak_y": int(peak_y), "peak_x": int(peak_x),
        "com_y": com_y, "com_x": com_x,
        "grid_pos": grid,
        "focused_area": focused_area,
        "entropy": entropy,
    }


def _grid_3x3(y: float, x: float, H: int, W: int) -> str:
    yi = 0 if y < H / 3 else (2 if y > 2 * H / 3 else 1)
    xi = 0 if x < W / 3 else (2 if x > 2 * W / 3 else 1)
    return [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]][yi][xi]


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
def denormalize(x: torch.Tensor) -> np.ndarray:
    """(C,H,W) normalizado -> (H,W,3) uint8."""
    arr = x.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Sobrepõe CAM (colormap jet) na imagem RGB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    H, W = img_rgb.shape[:2]
    cam_r = F.interpolate(
        torch.tensor(cam, dtype=torch.float32)[None, None],
        size=(H, W), mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    heat = (cm.jet(cam_r)[..., :3] * 255).astype(np.uint8)
    out = img_rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)