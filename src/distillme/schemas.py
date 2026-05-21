"""Typed data contracts shared by the distillation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

StageName = Literal["ingest", "investigate", "teach", "validate", "train", "evaluate"]


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one heterogeneous model role."""

    role: Literal["investigator", "teacher", "student"]
    family: str
    model: str
    endpoint: str = "local"
    max_context_tokens: int = 32768
    batch_size: int = 1


@dataclass(frozen=True)
class Artifact:
    """A source artifact discovered during repository ingestion."""

    artifact_id: str
    path: str
    kind: str
    language: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class Chunk:
    """An indexable source chunk that preserves evidence location."""

    chunk_id: str
    artifact_id: str
    path: str
    kind: str
    language: str
    start_line: int
    end_line: int
    text: str
    symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    """Relationship between indexed source entities."""

    source: str
    target: str
    relation: str
    evidence: str
    confidence: float


@dataclass(frozen=True)
class Finding:
    """Investigator finding with explicit uncertainty handling."""

    title: str
    category: str
    confidence: float
    supporting_evidence: tuple[str, ...]
    source_file_references: tuple[str, ...]
    inferred_reasoning: str
    counter_evidence: str = "None found in indexed artifacts."
    unresolved_ambiguity: str = "Requires deeper model-backed investigation."
    architectural_implications: str = "Documented for downstream teacher synthesis."

    def to_markdown(self) -> str:
        evidence = "\n".join(f"  - {item}" for item in self.supporting_evidence) or "  - None"
        refs = "\n".join(f"  - {item}" for item in self.source_file_references) or "  - None"
        return (
            "FINDING:\n"
            f"- title: {self.title}\n"
            f"- category: {self.category}\n"
            f"- confidence: {self.confidence:.2f}\n"
            "- supporting evidence:\n"
            f"{evidence}\n"
            "- source file references:\n"
            f"{refs}\n"
            f"- inferred reasoning: {self.inferred_reasoning}\n"
            f"- counter-evidence: {self.counter_evidence}\n"
            f"- unresolved ambiguity: {self.unresolved_ambiguity}\n"
            f"- architectural implications: {self.architectural_implications}\n"
        )


@dataclass(frozen=True)
class InvestigationIteration:
    """Captures one plan→execute iteration of the agentic investigation loop.

    Each iteration consists of two steps:

    1. **Plan** — the agent reviews accumulated evidence and decides which
       commands to run next, recording its *rationale*.
    2. **Execute** — the planned commands are run; new *findings* are extracted
       and the working *hypothesis* is updated.
    """

    iteration_num: int
    plan_rationale: str
    commands_planned: tuple[str, ...]
    findings: tuple[str, ...]
    hypothesis_after: str

    def to_markdown(self) -> str:
        cmds = "\n".join(f"    - `{c}`" for c in self.commands_planned) or "    - none"
        findings = "\n".join(f"    - {f}" for f in self.findings) or "    - none"
        return (
            f"#### Iteration {self.iteration_num + 1}\n\n"
            f"**Plan:** {self.plan_rationale}\n\n"
            f"**Commands:**\n{cmds}\n\n"
            f"**Findings:**\n{findings}\n\n"
            f"**Hypothesis after:** {self.hypothesis_after}\n"
        )


@dataclass(frozen=True)
class InvestigationTrace:
    """Structured investigative trace produced by the agentic investigation loop.

    Captures the full multi-iteration PLAN → EXECUTE cycle used by the
    :class:`AgenticInvestigatorLoop`.  Each iteration is stored in
    :attr:`iterations`.  The top-level fields aggregate the complete run.
    """

    objective: str
    hypothesis: str
    known_evidence: tuple[str, ...]
    uncertainties: tuple[str, ...]
    commands_run: tuple[str, ...]
    command_summaries: tuple[str, ...]
    updated_understanding: str
    next_investigation_step: str
    confidence: float
    iterations: tuple[InvestigationIteration, ...] = ()

    def to_markdown(self) -> str:
        evidence = "\n".join(f"  - {e}" for e in self.known_evidence) or "  - None gathered yet"
        uncerts = "\n".join(f"  - {u}" for u in self.uncertainties) or "  - None identified"
        commands = "\n".join(f"  - `{c}`" for c in self.commands_run) or "  - None executed"
        summaries = "\n".join(f"  - {s}" for s in self.command_summaries) or "  - No output"
        iter_section = ""
        if self.iterations:
            iter_bodies = "\n".join(it.to_markdown() for it in self.iterations)
            iter_section = f"**INVESTIGATION ITERATIONS ({len(self.iterations)}):**\n\n{iter_bodies}\n"
        return (
            "### Investigation Trace\n\n"
            f"**OBJECTIVE:** {self.objective}\n\n"
            f"**CURRENT HYPOTHESIS:** {self.hypothesis}\n\n"
            f"{iter_section}"
            "**KNOWN EVIDENCE:**\n"
            f"{evidence}\n\n"
            "**UNCERTAINTIES:**\n"
            f"{uncerts}\n\n"
            "**COMMANDS RUN:**\n"
            f"{commands}\n\n"
            "**COMMAND OUTPUT SUMMARY:**\n"
            f"{summaries}\n\n"
            f"**UPDATED UNDERSTANDING:** {self.updated_understanding}\n\n"
            f"**NEXT INVESTIGATION STEP:** {self.next_investigation_step}\n\n"
            f"**CONFIDENCE:** {self.confidence:.2f}\n"
        )


@dataclass(frozen=True)
class DatasetExample:
    """Synthetic instruction example emitted by the teacher stage."""

    task_id: str
    task_category: str
    difficulty: str
    repository_context: str
    retrieved_context: list[dict[str, Any]]
    question: str
    reasoning_trace: str
    answer: str
    supporting_files: list[str]
    symbols: list[str]
    architectural_constraints: list[str]
    validation_checks: list[str]
    negative_examples: list[str]
    confidence: float
    investigation_trace: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageResult:
    """Serializable stage execution metadata."""

    stage: StageName
    status: Literal["pending", "running", "succeeded", "failed"]
    outputs: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved filesystem layout for a pipeline run."""

    repository: Path
    workdir: Path
    index_dir: Path
    investigator_dir: Path
    dataset_dir: Path
    training_dir: Path
    evaluation_dir: Path
    logs_dir: Path

    @classmethod
    def from_root(cls, repository: Path, workdir: Path) -> "PipelinePaths":
        return cls(
            repository=repository,
            workdir=workdir,
            index_dir=workdir / "index",
            investigator_dir=workdir / "investigator",
            dataset_dir=workdir / "dataset",
            training_dir=workdir / "training",
            evaluation_dir=workdir / "evaluation",
            logs_dir=workdir / "logs",
        )

    def create(self) -> None:
        for path in asdict(self).values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
