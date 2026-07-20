"""
Shared text encoder — one encoder serves ALL adapters so every system searches
the same query embedding (the "same embeddings" fairness invariant).

Mirrors oracle/build_gt.py: google/siglip-base-patch16-224, text pooler output,
L2-normalised, 768-d, same space as the stored V3C image vectors.

⚠ LESSON (from the query-set repo): tokenise with padding='max_length' (64
tokens), NOT padding=True — the latter silently wrecks text↔image alignment and
targets rank ~random. Do not "simplify" this.

torch/transformers are an optional dependency (the `encode` extra); import lazily
so the light harness install doesn't need them.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

MODEL_NAME = "google/siglip-base-patch16-224"
MAX_LENGTH = 64


class Encoder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class SiglipEncoder:
    """Lazy-loaded SigLIP text encoder. Instantiate once, reuse across the run."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, text: str) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",      # ⚠ see module docstring — do NOT change
            max_length=MAX_LENGTH,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.text_model(**inputs).pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].cpu().numpy().astype(np.float32)


class CachingEncoder:
    """Wraps any Encoder with an in-memory text->vector cache (each item is
    encoded twice — vector_query and raw_query_text — and across systems the same
    texts recur, so caching avoids recomputation)."""

    def __init__(self, inner: Encoder):
        self.inner = inner
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is None:
            v = self.inner.encode(text)
            self._cache[text] = v
        return v
