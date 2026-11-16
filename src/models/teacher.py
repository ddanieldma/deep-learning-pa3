"""Wrapper único para foundation models usados como professor (Trilha C).

Backends suportados:
  * open_clip  -> BioCLIP, BioCLIP 2 (pesos abertos no HF Hub)
  * transformers -> CLIP, SigLIP

A interface é sempre a mesma:
  teacher.encode_images(images) -> (N, D) normalizado
  teacher.encode_texts(prompts) -> (P, D) normalizado
  teacher.class_embeddings(class_prompts) -> (K, D) normalizado
  teacher.classify_images(images, class_embs) -> (N, K) probabilidades
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registro de professores
# ---------------------------------------------------------------------------
# `candidates` existe porque a arquitetura exata do checkpoint do HF Hub
# pode variar; tentamos em ordem até uma carregar.
TEACHER_REGISTRY: dict[str, dict] = {
    "bioclip": {
        "backend": "open_clip",
        "hf_hub_id": "hf-hub:imageomics/bioclip",
    },
    "bioclip2": {
        "backend": "open_clip",
        "candidates": ["ViT-B-16", "ViT-L-14", "ViT-H-14"],
        "pretrained": "hf-hub:imageomics/bioclip-2",
    },
    "clip": {
        "backend": "transformers",
        "model_id": "openai/clip-vit-base-patch32",
    },
    "siglip": {
        "backend": "transformers",
        "model_id": "google/siglip-base-patch16-224",
    },
}


class TeacherModel:
    """Professor com interface única para open_clip e transformers."""

    def __init__(
        self,
        name: str = "bioclip2",
        device: str | None = None,
        model_id: str | None = None,
        pretrained: str | None = None,
        open_clip_model: str | None = None,
    ) -> None:
        self.name = name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if name not in TEACHER_REGISTRY:
            raise ValueError(
                f"Professor desconhecido: {name}. "
                f"Opções: {list(TEACHER_REGISTRY)}"
            )

        spec = dict(TEACHER_REGISTRY[name])
        if model_id is not None:
            spec["model_id"] = model_id
        if pretrained is not None:
            spec["pretrained"] = pretrained
        if open_clip_model is not None:
            spec["candidates"] = [open_clip_model]

        self.backend = spec["backend"]
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self._class_emb_cache: dict[tuple[tuple[str, ...], ...], torch.Tensor] = {}

        if self.backend == "open_clip":
            self._load_open_clip(spec)
        elif self.backend == "transformers":
            self._load_transformers(spec)
        else:
            raise ValueError(f"Backend não suportado: {self.backend}")

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------
    def _load_open_clip(self, spec: dict) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Instale `open_clip_torch`: pip install -U open_clip_torch"
            ) from exc

        hf_hub_id = spec.get("hf_hub_id") or spec.get("pretrained")
        if not hf_hub_id or not hf_hub_id.startswith("hf-hub:"):
            raise ValueError(
                f"Para open_clip, esperava um identificador 'hf-hub:...', recebi {hf_hub_id!r}"
            )

        try:
            # Passa o identificador hf-hub como model_name; NÃO passa pretrained separado.
            model, _, preprocess = open_clip.create_model_and_transforms(hf_hub_id)
            model = model.to(self.device).eval()
            tokenizer = open_clip.get_tokenizer(hf_hub_id)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Falha ao carregar {hf_hub_id} via open_clip. "
                "Verifique se `open_clip_torch` está atualizado e se o repo existe no HF Hub."
            ) from exc

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        logger.info("Carregado open_clip %s em %s", hf_hub_id, self.device)

    def _load_transformers(self, spec: dict) -> None:
        from transformers import AutoModel, AutoProcessor

        model_id = spec["model_id"]
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        logger.info("Carregado transformers %s em %s", model_id, self.device)

    # ------------------------------------------------------------------
    # Pré-processamento
    # ------------------------------------------------------------------
    def preprocess_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        images = [im.convert("RGB") for im in images]

        if self.backend == "open_clip":
            return torch.stack([self.preprocess(im) for im in images])

        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        pixel_values = self.preprocess_images(images).to(self.device)

        if self.backend == "open_clip":
            feats = self.model.encode_image(pixel_values)
        else:
            feats = self.model.get_image_features(pixel_values=pixel_values)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output

        return F.normalize(feats, dim=-1)


    @torch.no_grad()
    def encode_texts(self, prompts: Sequence[str]) -> torch.Tensor:
        key = tuple(prompts)
        if key in self._text_cache:
            return self._text_cache[key]

        if self.backend == "open_clip":
            tokens = self.tokenizer(list(prompts)).to(self.device)
            feats = self.model.encode_text(tokens)
        else:
            inputs = self.processor(
                text=list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            feats = self.model.get_text_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output

        feats = F.normalize(feats, dim=-1)
        self._text_cache[key] = feats
        return feats

    # ------------------------------------------------------------------
    # Embeddings de classe (ensemble de prompts)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def class_embeddings(
        self, class_prompts: Sequence[Sequence[str]]
    ) -> torch.Tensor:
        """Média normalizada dos embeddings dos prompts de cada classe."""
        key = tuple(tuple(p) for p in class_prompts)
        if key in self._class_emb_cache:
            return self._class_emb_cache[key]

        # Flatten + índice prompt -> posição.
        prompt_to_idx: dict[str, int] = {}
        for prompts in class_prompts:
            for p in prompts:
                if p not in prompt_to_idx:
                    prompt_to_idx[p] = len(prompt_to_idx)

        flat_prompts = list(prompt_to_idx.keys())
        index = [
            [prompt_to_idx[p] for p in prompts] for prompts in class_prompts
        ]

        text_embs = self.encode_texts(flat_prompts)  # (P, D)

        class_embs = []
        for idxs in index:
            e = text_embs[idxs].mean(dim=0)
            class_embs.append(e)

        class_embs = F.normalize(torch.stack(class_embs, dim=0), dim=-1)
        self._class_emb_cache[key] = class_embs
        return class_embs

    # ------------------------------------------------------------------
    # Classificação
    # ------------------------------------------------------------------
    @torch.no_grad()
    def classify_images(
        self,
        images: Sequence[Image.Image],
        class_embeddings: torch.Tensor,
        temperature: float = 0.01,
    ) -> torch.Tensor:
        """Retorna `(N, K)` probabilidades softmax."""
        img_embs = self.encode_images(images)  # (N, D)
        logits = img_embs @ class_embeddings.T / temperature
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def classify(
        self,
        images: Sequence[Image.Image],
        class_prompts: Sequence[Sequence[str]],
        temperature: float = 0.01,
    ) -> torch.Tensor:
        class_embs = self.class_embeddings(class_prompts)
        return self.classify_images(images, class_embs, temperature=temperature)