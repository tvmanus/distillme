"""Hybrid retrieval over local indexes with dense, sparse, and symbol signals.

Provides two retrieval backends:

* :class:`HybridRetriever` — fully deterministic, no external dependencies;
  combines hash-based dense projection, BM25 approximate sparse scoring, and
  symbol overlap.

* :class:`ChromaRetriever` — ChromaDB-backed vector store with real embedding
  support via :class:`~distillme.embedding.EmbeddingClient`.  Requires
  ``chromadb`` (``pip install 'distillme[chroma]'``).  Chunks are indexed
  lazily on the first query by reading the JSONL produced by the ingest stage.

Use :func:`make_retriever` to obtain the appropriate backend based on the
configured :attr:`~distillme.config.RetrievalConfig.vector_backend`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from distillme.schemas import Chunk

if TYPE_CHECKING:
    from distillme.config import RetrievalConfig
    from distillme.embedding import EmbeddingClient

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


# ---------------------------------------------------------------------------
# ChromaDB-backed vector retriever
# ---------------------------------------------------------------------------


class ChromaRetriever:
    """ChromaDB-backed vector retriever with real embedding support.

    Chunks produced by the ingest stage (written to ``chunks.jsonl``) are
    indexed into a persistent ChromaDB collection on the *first* call to
    :meth:`search`.  Subsequent calls reuse the stored collection, making
    resume-safe across pipeline runs.

    Requires ``chromadb`` to be installed::

        pip install 'distillme[chroma]'

    Parameters
    ----------
    index_dir:
        Directory that contains the ``chunks.jsonl`` written by the ingest
        stage.  The ChromaDB data is stored in a ``chroma/`` subdirectory.
    embedding_client:
        Client used to produce embedding vectors for both indexing and queries.
    collection_name:
        Name of the ChromaDB collection.  Change this only when running
        multiple independent indexes in the same workdir.
    """

    def __init__(
        self,
        index_dir: Path,
        embedding_client: "EmbeddingClient",
        collection_name: str = "distillme_chunks",
    ) -> None:
        self.index_dir = index_dir
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        self._chroma_client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        """Return the *top_k* most relevant chunks for *query*."""
        self._ensure_indexed()
        count = self._collection.count()
        if count == 0:
            return []
        n = min(top_k, count)
        results = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievalHit] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns L2 or cosine distances; convert to a 0-1 similarity.
            score = max(0.0, 1.0 - float(dist))
            chunk = Chunk(
                chunk_id=meta.get("chunk_id", ""),
                artifact_id=meta.get("artifact_id", ""),
                path=meta["path"],
                kind=meta["kind"],
                language=meta["language"],
                start_line=int(meta["start_line"]),
                end_line=int(meta["end_line"]),
                text=doc,
                symbols=tuple(s for s in meta.get("symbols", "").split(",") if s),
            )
            hits.append(RetrievalHit(chunk=chunk, score=score, reasons=("vector_similarity",)))
        return hits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_indexed(self) -> None:
        """Lazily initialise and populate the ChromaDB collection."""
        if self._collection is not None:
            return
        try:
            import chromadb  # type: ignore[import]
            from chromadb import EmbeddingFunction, Documents, Embeddings  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ChromaDB is required for vector_backend='chroma'. "
                "Install it with: pip install 'distillme[chroma]'"
            ) from exc

        embedding_client = self.embedding_client

        class _DistillmeEmbeddingFunction(EmbeddingFunction[Documents]):  # type: ignore[misc]
            def __init__(self) -> None:
                # ChromaDB ≥1.5 warns if __init__ is not overridden; override
                # here to silence that deprecation without forwarding to the
                # base-class stub that triggers the warning.
                pass

            @staticmethod
            def name() -> str:
                return "distillme"

            def get_config(self) -> dict:  # type: ignore[override]
                return {"name": "distillme"}

            @classmethod
            def build_from_config(cls, config: dict) -> "_DistillmeEmbeddingFunction":  # type: ignore[override]
                return cls()

            def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]  # noqa: A002
                return embedding_client.embed(list(input))

        persist_dir = self.index_dir / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=_DistillmeEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        if self._collection.count() == 0:
            self._populate_from_jsonl()

    def _populate_from_jsonl(self) -> None:
        """Read ``chunks.jsonl`` and add all chunks to the collection in batches."""
        chunks = _load_chunks(self.index_dir / "chunks.jsonl")
        if not chunks:
            return
        batch_size = 100
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            self._collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {
                        "chunk_id": chunk.chunk_id,
                        "artifact_id": chunk.artifact_id,
                        "path": chunk.path,
                        "kind": chunk.kind,
                        "language": chunk.language,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "symbols": ",".join(chunk.symbols),
                    }
                    for chunk in batch
                ],
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_retriever(
    retrieval_config: "RetrievalConfig",
    index_dir: Path,
) -> "HybridRetriever | ChromaRetriever":
    """Return the retrieval backend specified by *retrieval_config*.

    * ``vector_backend = "local-jsonl"`` (default) → :class:`HybridRetriever`
    * ``vector_backend = "chroma"`` → :class:`ChromaRetriever`

    The embedding client for the Chroma backend is constructed from the
    ``embedding_endpoint``, ``embedding_model``, and ``embedding_api_key``
    fields in *retrieval_config* via
    :func:`~distillme.embedding.make_embedding_client`.
    """
    if retrieval_config.vector_backend == "chroma":
        from distillme.embedding import make_embedding_client

        embedding_client = make_embedding_client(
            endpoint=retrieval_config.embedding_endpoint,
            model=retrieval_config.embedding_model,
            api_key=retrieval_config.embedding_api_key,
        )
        return ChromaRetriever(index_dir=index_dir, embedding_client=embedding_client)

    return HybridRetriever(
        index_dir=index_dir,
        dense_weight=retrieval_config.dense_weight,
        sparse_weight=retrieval_config.sparse_weight,
        symbol_weight=retrieval_config.symbol_weight,
    )

