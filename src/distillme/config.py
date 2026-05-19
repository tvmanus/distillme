"""Configuration loading and heterogeneity validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distillme.schemas import ModelSpec


@dataclass(frozen=True)
class RetrievalConfig:
    vector_backend: str = "local-jsonl"
    dense_weight: float = 0.45
    sparse_weight: float = 0.35
    symbol_weight: float = 0.20
    max_chunk_lines: int = 120
    top_k: int = 8


@dataclass(frozen=True)
class PipelineConfig:
    repository_path: Path
    workdir: Path
    investigator: ModelSpec
    teacher: ModelSpec
    student: ModelSpec
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    include_globs: tuple[str, ...] = ("**/*",)
    exclude_dirs: tuple[str, ...] = (
        ".git",
        ".gradle",
        ".idea",
        ".mvn/wrapper",
        "build",
        "target",
        "node_modules",
        "dist",
        "__pycache__",
    )

    @classmethod
    def default(cls, repository_path: Path, workdir: Path) -> "PipelineConfig":
        return cls(
            repository_path=repository_path,
            workdir=workdir,
            investigator=ModelSpec(
                role="investigator",
                family="gemini",
                model="gemini-1.5-pro-or-compatible",
                max_context_tokens=1_000_000,
            ),
            teacher=ModelSpec(
                role="teacher",
                family="claude",
                model="claude-3.5-sonnet-or-compatible",
                max_context_tokens=200_000,
            ),
            student=ModelSpec(
                role="student",
                family="qwen2.5",
                model="Qwen2.5-Coder-7B-Instruct",
                max_context_tokens=32768,
                batch_size=8,
            ),
        )

    @classmethod
    def from_file(cls, path: Path) -> "PipelineConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PipelineConfig":
        retrieval_raw = raw.get("retrieval", {})
        return cls(
            repository_path=Path(raw["repository_path"]).expanduser().resolve(),
            workdir=Path(raw["workdir"]).expanduser().resolve(),
            investigator=ModelSpec(role="investigator", **raw["models"]["investigator"]),
            teacher=ModelSpec(role="teacher", **raw["models"]["teacher"]),
            student=ModelSpec(role="student", **raw["models"]["student"]),
            retrieval=RetrievalConfig(**retrieval_raw),
            include_globs=tuple(raw.get("include_globs", ("**/*",))),
            exclude_dirs=tuple(raw.get("exclude_dirs", cls.__dataclass_fields__["exclude_dirs"].default)),
        )

    def validate(self) -> None:
        families = {self.investigator.family, self.teacher.family, self.student.family}
        if len(families) != 3:
            raise ValueError("investigator, teacher, and student model families must be distinct")
        if not self.repository_path.exists():
            raise FileNotFoundError(f"repository_path does not exist: {self.repository_path}")
        if self.retrieval.top_k < 1:
            raise ValueError("retrieval.top_k must be positive")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "repository_path": str(self.repository_path),
            "workdir": str(self.workdir),
            "models": {
                "investigator": _model_to_json(self.investigator),
                "teacher": _model_to_json(self.teacher),
                "student": _model_to_json(self.student),
            },
            "retrieval": self.retrieval.__dict__,
            "include_globs": list(self.include_globs),
            "exclude_dirs": list(self.exclude_dirs),
        }


def _model_to_json(model: ModelSpec) -> dict[str, Any]:
    return {
        "family": model.family,
        "model": model.model,
        "endpoint": model.endpoint,
        "max_context_tokens": model.max_context_tokens,
        "batch_size": model.batch_size,
    }
