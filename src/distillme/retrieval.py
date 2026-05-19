"""Hybrid retrieval over local indexes with dense, sparse, and symbol signals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from distillme.schemas import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    reasons: tuple[str, ...]

    def to_context(self) -> dict[str, object]:
        return {
            "path": self.chunk.path,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "symbols": list(self.chunk.symbols),
            "score": round(self.score, 4),
            "text": self.chunk.text,
            "reasons": list(self.reasons),
        }


class HybridRetriever:
    """Deterministic retrieval backend suitable for local validation and tests."""

    def __init__(
        self,
        index_dir: Path,
        dense_weight: float = 0.45,
        sparse_weight: float = 0.35,
        symbol_weight: float = 0.20,
    ) -> None:
        self.index_dir = index_dir
        self.chunks = _load_chunks(index_dir / "chunks.jsonl")
        self.document_frequency = _document_frequency(self.chunks)
        self.chunk_tokens = {chunk.chunk_id: _tokens(chunk.text) for chunk in self.chunks}
        self.term_frequencies = {chunk.chunk_id: _term_counts(self.chunk_tokens[chunk.chunk_id]) for chunk in self.chunks}
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.symbol_weight = symbol_weight

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        query_tokens = set(_tokens(query))
        query_vector = _hash_vector(query_tokens)
        hits: list[RetrievalHit] = []
        for chunk in self.chunks:
            chunk_tokens = self.chunk_tokens[chunk.chunk_id]
            sparse = _bm25_approximate_score(
                query_tokens,
                self.term_frequencies[chunk.chunk_id],
                self.document_frequency,
                max(len(self.chunks), 1),
            )
            dense = _cosine(query_vector, _hash_vector(chunk_tokens))
            symbol_match_count = len(query_tokens.intersection({symbol.lower() for symbol in chunk.symbols}))
            score = self.dense_weight * dense + self.sparse_weight * sparse + self.symbol_weight * symbol_match_count
            if score > 0:
                reasons = []
                if dense > 0:
                    reasons.append("dense_hash_overlap")
                if sparse > 0:
                    reasons.append("bm25_token_overlap")
                if symbol_match_count > 0:
                    reasons.append("symbol_match")
                hits.append(RetrievalHit(chunk=chunk, score=score, reasons=tuple(reasons)))
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]


def _load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    chunks = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            raw["symbols"] = tuple(raw.get("symbols", ()))
            chunks.append(Chunk(**raw))
    return chunks


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _document_frequency(chunks: Iterable[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        for token in set(_tokens(chunk.text)):
            counts[token] = counts.get(token, 0) + 1
    return counts


def _bm25_approximate_score(
    query_tokens: set[str], term_counts: dict[str, int], document_frequency: dict[str, int], documents: int
) -> float:
    if not query_tokens or not term_counts:
        return 0.0
    score = 0.0
    for token in query_tokens:
        tf = term_counts.get(token, 0)
        if not tf:
            continue
        idf = math.log((documents + 1) / (document_frequency.get(token, 0) + 1)) + 1
        score += idf * (tf / (tf + 1.2))
    return score / max(len(query_tokens), 1)


def _term_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _hash_vector(tokens: Iterable[str], dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    return vector


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(numerator / (left_norm * right_norm), 0.0)
