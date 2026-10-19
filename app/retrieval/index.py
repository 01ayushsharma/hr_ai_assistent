"""Hybrid lexical + dense index over chunks (in-process; the corpus is small).

BM25 catches exact policy terms ("carry forward", "Ugadi"); dense embeddings
catch paraphrases ("doctor's note" vs "medical certificate"). Scores are fused
with reciprocal-rank fusion, so neither scale needs calibrating. Search always
runs over a caller-supplied candidate set: eligibility filtering happens before
any scoring.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

import numpy as np

from ..logging_setup import log_event

logger = logging.getLogger(__name__)

_STOP = set(
    """a an and are as at be by can do does for from get got has have how i if in into is it its
    me my of on or our so than that the their them then there these they this to too was we
    what when where which who will with you your am been being did done shall should would
    could may might must about any all also per each""".split()
)
_RRF_K = 60
_MIN_COSINE = 0.62


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], i: int) -> float:
        tf, length, total = self.tf[i], self.len[i], 0.0
        for term in set(query):
            f = tf.get(term)
            if not f:
                continue
            norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            total += self.idf.get(term, 0.0) * norm
        return total


class Embedder:
    """Local CPU embedding model (fastembed / ONNX)."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(list(self.model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.array(list(self.model.query_embed([text])), dtype=np.float32))[0]


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-9, None)


def load_embedder(model_name: str, cache_dir: str) -> Embedder | None:
    try:
        return Embedder(model_name, cache_dir)
    except Exception as exc:  # network or model problems must not stop the service
        log_event(logger, "embedder_unavailable", logging.WARNING, error=type(exc).__name__,
                  detail="falling back to BM25-only retrieval")
        return None


class HybridIndex:
    def __init__(self, texts: list[str], embedder: Embedder | None) -> None:
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.embedder = embedder
        self.vectors = embedder.embed_documents(texts) if embedder and texts else None

    @property
    def dense_enabled(self) -> bool:
        return self.vectors is not None

    def search(self, query: str, candidates: list[int], top_k: int) -> list[tuple[int, float]]:
        """Return (chunk index, fused score) for the best candidates, best first."""
        if not candidates:
            return []
        q_tokens = tokenize(query)
        lexical = {i: self.bm25.score(q_tokens, i) for i in candidates}
        dense: dict[int, float] = {}
        if self.vectors is not None:
            q = self.embedder.embed_query(query)
            sims = self.vectors[candidates] @ q
            dense = {i: float(s) for i, s in zip(candidates, sims)}

        relevant = [i for i in candidates if lexical[i] > 0 or dense.get(i, 0.0) >= _MIN_COSINE]
        fused: dict[int, float] = {i: 0.0 for i in relevant}
        for scores in (lexical, dense):
            if not scores:
                continue
            ranked = sorted(relevant, key=lambda i: (-scores[i], i))
            for rank, i in enumerate(ranked):
                if scores[i] > 0:
                    fused[i] += 1.0 / (_RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
