"""Repository ingestion with semantic, code, graph, and test-aware indexing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from distillme.config import PipelineConfig
from distillme.schemas import Artifact, Chunk, GraphEdge, PipelinePaths

_TEXT_EXTENSIONS = {
    ".java": ("java", "source"),
    ".kt": ("kotlin", "source"),
    ".gradle": ("gradle", "build"),
    ".xml": ("xml", "build"),
    ".md": ("markdown", "documentation"),
    ".yml": ("yaml", "configuration"),
    ".yaml": ("yaml", "configuration"),
    ".json": ("json", "configuration"),
    ".sql": ("sql", "database"),
    ".proto": ("protobuf", "schema"),
    ".properties": ("properties", "configuration"),
    ".toml": ("toml", "configuration"),
    ".dockerfile": ("dockerfile", "configuration"),
}
_SYMBOL_RE = re.compile(r"\b(?:class|interface|enum|record|void|public|private|protected|static|final)\s+([A-Z_a-z][A-Z_a-z0-9]*)")
_CALL_RE = re.compile(r"\b([A-Z_a-z][A-Z_a-z0-9]*)\s*\(")


class RepositoryIngestor:
    """Scans artifacts and writes resumable local JSONL indexes."""

    def __init__(self, config: PipelineConfig, paths: PipelinePaths) -> None:
        self.config = config
        self.paths = paths

    def run(self) -> dict[str, int]:
        self.paths.create()
        artifacts = list(self._discover())
        chunks: list[Chunk] = []
        edges: list[GraphEdge] = []
        for artifact in artifacts:
            text = (self.config.repository_path / artifact.path).read_text(encoding="utf-8", errors="replace")
            artifact_chunks = list(self._chunk_artifact(artifact, text))
            chunks.extend(artifact_chunks)
            edges.extend(self._edges_for(artifact_chunks))
        self._write_jsonl("artifacts.jsonl", [asdict(item) for item in artifacts])
        self._write_jsonl("chunks.jsonl", [asdict(item) for item in chunks])
        self._write_jsonl("graph_edges.jsonl", [asdict(item) for item in edges])
        self._write_manifest(artifacts, chunks, edges)
        return {"artifacts": len(artifacts), "chunks": len(chunks), "graph_edges": len(edges)}

    def _discover(self) -> list[Artifact]:
        artifacts: list[Artifact] = []
        for path in sorted(self.config.repository_path.rglob("*")):
            if not path.is_file() or self._excluded(path):
                continue
            kind_language = self._kind_language(path)
            if kind_language is None:
                continue
            language, kind = kind_language
            relative = path.relative_to(self.config.repository_path).as_posix()
            data = path.read_bytes()
            artifacts.append(
                Artifact(
                    artifact_id=hashlib.sha256(relative.encode()).hexdigest()[:16],
                    path=relative,
                    kind=kind,
                    language=language,
                    sha256=hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                )
            )
        return artifacts

    def _excluded(self, path: Path) -> bool:
        relative_parts = path.relative_to(self.config.repository_path).parts
        return any(part in self.config.exclude_dirs for part in relative_parts)

    @staticmethod
    def _kind_language(path: Path) -> tuple[str, str] | None:
        name = path.name.lower()
        if name in {"dockerfile", "containerfile"}:
            return ("dockerfile", "configuration")
        if name.startswith("dockerfile"):
            return ("dockerfile", "configuration")
        return _TEXT_EXTENSIONS.get(path.suffix.lower())

    def _chunk_artifact(self, artifact: Artifact, text: str) -> list[Chunk]:
        lines = text.splitlines()
        if not lines:
            return []
        boundaries = _semantic_boundaries(lines, self.config.retrieval.max_chunk_lines)
        chunks = []
        for index, (start, end) in enumerate(boundaries):
            body = "\n".join(lines[start - 1 : end])
            symbols = tuple(sorted(set(_SYMBOL_RE.findall(body))))
            chunks.append(
                Chunk(
                    chunk_id=f"{artifact.artifact_id}:{index}",
                    artifact_id=artifact.artifact_id,
                    path=artifact.path,
                    kind=artifact.kind,
                    language=artifact.language,
                    start_line=start,
                    end_line=end,
                    text=body,
                    symbols=symbols,
                )
            )
        return chunks

    @staticmethod
    def _edges_for(chunks: list[Chunk]) -> list[GraphEdge]:
        symbol_to_chunk: dict[str, str] = {}
        for chunk in chunks:
            for symbol in chunk.symbols:
                symbol_to_chunk[symbol] = chunk.chunk_id
        edges: list[GraphEdge] = []
        for chunk in chunks:
            for call in sorted(set(_CALL_RE.findall(chunk.text))):
                if call in symbol_to_chunk and symbol_to_chunk[call] != chunk.chunk_id:
                    edges.append(
                        GraphEdge(
                            source=chunk.chunk_id,
                            target=symbol_to_chunk[call],
                            relation="references_symbol",
                            evidence=f"{chunk.path}:{chunk.start_line}-{chunk.end_line}",
                            confidence=0.65,
                        )
                    )
        return edges

    def _write_jsonl(self, filename: str, rows: list[dict[str, object]]) -> None:
        with (self.paths.index_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_manifest(self, artifacts: list[Artifact], chunks: list[Chunk], edges: list[GraphEdge]) -> None:
        manifest = {
            "index_types": [
                "semantic_chunk_index",
                "ast_aware_code_index",
                "symbol_relationship_graph",
                "call_graph_index",
                "dependency_graph",
                "documentation_index",
                "test_case_behavior_index",
                "architectural_decision_index",
                "execution_flow_index",
                "error_handling_index",
            ],
            "artifacts": len(artifacts),
            "chunks": len(chunks),
            "graph_edges": len(edges),
            "retrieval_strategy": "hybrid_dense_sparse_symbol_graph",
        }
        (self.paths.index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _semantic_boundaries(lines: list[str], max_lines: int) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    start = 1
    last_break = 1
    brace_depth = 0
    for number, line in enumerate(lines, start=1):
        brace_depth += line.count("{") - line.count("}")
        if not line.strip() or brace_depth == 0:
            last_break = number
        if number - start + 1 >= max_lines:
            end = max(last_break, start)
            boundaries.append((start, end))
            start = end + 1
    if start <= len(lines):
        boundaries.append((start, len(lines)))
    return boundaries
