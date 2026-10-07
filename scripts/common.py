"""Shared utilities for C-Mining scripts.

Keeps embedding-model loading, text formatting, geometric metrics
(Local Dispersion, Semantic Entropy) and JSONL I/O in one place so that
step0/step1/step2 stay thin.
"""
import json
import os
import sys
from typing import Any, Iterable, List

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EMBED_BATCH_SIZE, EMBED_DEVICE, EMBED_MODEL, KNN_K

_EMBEDDER: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Lazy singleton for the multilingual encoder (loaded once per process)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _EMBEDDER


def encode(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    return get_embedder().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def first_paragraph(text: str) -> str:
    if not text:
        return ""
    return text.split("\n")[0].strip()


def format_input(title: str, text: str) -> str:
    """Paper §3.2: S = '<title> {T} <text> {P1}'."""
    return f"<title> {title.strip()} <text> {first_paragraph(text)}"


def read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n


def local_dispersion(embeddings: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Local Dispersion δ_m: mean Euclidean distance to k-nearest neighbors (Eq. 1)."""
    n = len(embeddings)
    if n <= k:
        return np.zeros(n, dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(embeddings)
    dists, _ = nn.kneighbors(embeddings)
    return dists.mean(axis=1).astype(np.float32)


def entropy_of_similarity(emb: np.ndarray) -> float:
    """Semantic Entropy H(D): mean Shannon entropy over row-normalized
    paragraph-cosine-similarity matrix (Eq. 2-3)."""
    if len(emb) < 2:
        return 0.0
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = normed @ normed.T
    prob = sim / (sim.sum(axis=1, keepdims=True) + 1e-8)
    prob = np.clip(prob, 1e-8, 1.0)
    return float(np.mean(scipy_entropy(prob, axis=1)))


def get_kmeans(n_clusters: int):
    """cuML KMeans with a sklearn fallback so the pipeline runs on CPU too."""
    try:
        from cuml import KMeans
        return KMeans(n_clusters=n_clusters, random_state=42)
    except ImportError:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
