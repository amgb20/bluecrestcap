"""Deterministic local embedding utilities for offline operation."""
from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np


DEFAULT_EMBEDDING_DIM = 1536


def _seed_from_text(text: str) -> int:
    """Create a stable seed value from text content."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)


def generate_fake_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> List[float]:
    """Produce a deterministic unit-norm embedding for the provided text."""
    rng = np.random.default_rng(_seed_from_text(text))
    vector = rng.standard_normal(dim)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return [0.0] * dim
    return (vector / norm).astype(float).tolist()


def generate_fake_embeddings(texts: Iterable[str], dim: int = DEFAULT_EMBEDDING_DIM) -> List[List[float]]:
    """Generate embeddings for an iterable of texts."""
    return [generate_fake_embedding(text, dim=dim) for text in texts]
