"""Semantic embedding module using sentence-transformers.

Provides pre-encoding via fit() and cached retrieval via get_embedding().
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_cache: dict[str, np.ndarray] = {}


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded sentence-transformer model: all-MiniLM-L6-v2")
    return _model


def fit(texts: list[str]) -> None:
    """Pre-encode a corpus of texts for fast cached retrieval."""
    model = _load_model()
    _cache.clear()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    for text, vec in zip(texts, vectors):
        _cache[text] = vec
    logger.info("Pre-encoded %d texts (%d-dim embeddings)", len(texts), vectors.shape[1])


def get_embedding(text: str) -> np.ndarray:
    if text in _cache:
        return _cache[text]
    model = _load_model()
    vec = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    _cache[text] = vec
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def reset() -> None:
    _cache.clear()
