"""Investigator stage that emits evidence-backed architecture documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from distillme.retrieval import RetrievalHit
from distillme.retrieval import HybridRetriever
from distillme.schemas import Finding, InvestigationTrace, PipelinePaths

if TYPE_CHECKING:
    from distillme.cli_tools import CliExecutor, CliResult
    from distillme.inference import LLMClient

_INVESTIGATOR_SYSTEM = (
    "You are an expert software archaeologist performing deep codebase analysis. "
    "All claims must be grounded in the retrieved source evidence provided. "
    "Mark uncertainty explicitly. Do not fabricate class names, APIs, or runtime behavior."
)

MANDATORY_DOCUMENTS = (
    "architecture_overview.md",
    "service_catalog.md",
    "package_relationships.md",
    "domain_model.md",
    "coding_conventions.md",
    "testing_strategy.md",
    "algorithms_catalog.md",
    "concurrency_patterns.md",
    "security_analysis.md",
    "dependency_analysis.md",
    "api_behavior.md",
    "database_model.md",
    "execution_flows.md",
    "anti_patterns.md",
    "performance_characteristics.md",
    "operational_behavior.md",
    "configuration_model.md",
    "event_flow_analysis.md",
    "state_machine_analysis.md",
    "caching_behavior.md",
    "exception_taxonomy.md",
    "glossary.md",
)

_DOCUMENT_QUERIES = {
    "architecture_overview.md": "architecture service module package dependency entrypoint",
    "service_catalog.md": "service controller application endpoint component",
    "package_relationships.md": "package import dependency module",
    "domain_model.md": "entity domain model aggregate invariant workflow",
    "coding_conventions.md": "class method naming error style convention",
    "testing_strategy.md": "test junit assert integration unit",
    "algorithms_catalog.md": "algorithm cache optimize compute complexity",
    "concurrency_patterns.md": "thread async executor lock synchronized reactive",
    "security_analysis.md": "auth authorization validation secret injection crypto",
    "dependency_analysis.md": "gradle maven dependency plugin version",
    "api_behavior.md": "api controller request response endpoint",
    "database_model.md": "sql repository transaction migration schema",
    "execution_flows.md": "flow handler process execute call",
    "anti_patterns.md": "todo deprecated fixme workaround static global",
    "performance_characteristics.md": "cache latency memory allocation query hot path",
    "operational_behavior.md": "docker kubernetes config health metrics logging",
    "configuration_model.md": "property yaml config environment feature flag",
    "event_flow_analysis.md": "event listener publish subscribe queue message",
    "state_machine_analysis.md": "state enum transition status lifecycle",
    "caching_behavior.md": "cache memoize ttl eviction",
    "exception_taxonomy.md": "exception error throw catch retry failure",
    "glossary.md": "class interface enum domain terminology",
}

# CLI exploration commands executed per document topic during agentic investigation.
# Each entry is an ordered list of argument vectors (no shell expansion).
_TOPIC_CLI_COMMANDS: dict[str, list[list[str]]] = {
    "architecture_overview.md": [
        ["find", ".", "-name", "*.java", "-type", "f"],
        ["find", ".", "-name", "build.gradle", "-type", "f"],
        ["find", ".", "-name", "pom.xml", "-type", "f"],
    ],
    "service_catalog.md": [
        ["grep", "-r", "--include=*.java", "-l", "Service", "."],
        ["grep", "-r", "--include=*.java", "-l", "Controller", "."],
        ["grep", "-r", "--include=*.java", "-l", "Component", "."],
    ],
    "package_relationships.md": [
        ["grep", "-r", "--include=*.java", "-h", "^package ", "."],
        ["grep", "-r", "--include=*.java", "-h", "^import ", "."],
    ],
    "domain_model.md": [
        ["grep", "-r", "--include=*.java", "-l", "Entity", "."],
        ["grep", "-r", "--include=*.java", "-rn", "class.*implements", "."],
    ],
    "testing_strategy.md": [
        ["find", ".", "-path", "*/test/*", "-name", "*.java", "-type", "f"],
        ["grep", "-r", "--include=*.java", "-l", "Test", "."],
    ],
    "dependency_analysis.md": [
        ["find", ".", "-name", "build.gradle", "-type", "f"],
        ["find", ".", "-name", "pom.xml", "-type", "f"],
        ["find", ".", "-name", "*.toml", "-type", "f"],
    ],
    "execution_flows.md": [
        ["grep", "-r", "--include=*.java", "-n", "void main", "."],
        ["grep", "-r", "--include=*.java", "-l", "Handler", "."],
    ],
    "concurrency_patterns.md": [
        ["grep", "-r", "--include=*.java", "-l", "synchronized", "."],
        ["grep", "-r", "--include=*.java", "-l", "Executor", "."],
        ["grep", "-r", "--include=*.java", "-l", "Thread", "."],
    ],
    "security_analysis.md": [
        ["grep", "-r", "--include=*.java", "-l", "Security", "."],
        ["grep", "-r", "--include=*.java", "-l", "Auth", "."],
    ],
    "event_flow_analysis.md": [
        ["grep", "-r", "--include=*.java", "-l", "EventListener", "."],
        ["grep", "-r", "--include=*.java", "-l", "Publisher", "."],
    ],
    "exception_taxonomy.md": [
        ["grep", "-r", "--include=*.java", "-l", "Exception", "."],
        ["grep", "-r", "--include=*.java", "-n", "throws", "."],
    ],
    "caching_behavior.md": [
        ["grep", "-r", "--include=*.java", "-l", "Cache", "."],
        ["grep", "-r", "--include=*.java", "-l", "Cacheable", "."],
    ],
}

# Default CLI commands used for topics not explicitly mapped above.
_DEFAULT_CLI_COMMANDS: list[list[str]] = [
    ["find", ".", "-name", "*.java", "-type", "f"],
    ["find", ".", "-name", "*.gradle", "-type", "f"],
]


class InvestigationMemory:
    """Mutable working-memory accumulator for one agentic investigation pass.

    Records CLI commands executed, their output summaries, accumulated evidence
    snippets, uncertainties, and a continuously-refined hypothesis.  Call
    :meth:`to_trace` to produce the immutable :class:`InvestigationTrace` that
    is embedded in the generated document.
    """

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.hypothesis: str = f"The codebase contains {topic.removesuffix('.md').replace('_', ' ')} patterns that can be identified through systematic source exploration."
        self.evidence: list[str] = []
        self.commands_run: list[str] = []
        self.command_summaries: list[str] = []
        self.uncertainties: list[str] = [
            "Static analysis alone cannot confirm runtime behaviour.",
            "Indexed artifacts may not represent the full deployed codebase.",
        ]
        self.updated_understanding: str = ""

    def record_command(self, result: "CliResult", summary: str) -> None:
        self.commands_run.append(result.command)
        self.command_summaries.append(summary)

    def add_evidence(self, evidence: str) -> None:
        self.evidence.append(evidence)

    def refine_hypothesis(self, new_hypothesis: str) -> None:
        self.hypothesis = new_hypothesis

    def to_trace(self, objective: str, confidence: float) -> InvestigationTrace:
        understanding = self.updated_understanding or (
            f"Completed CLI-assisted investigation of {self.topic}. "
            "Findings are grounded in executed command output and RAG-retrieved chunks."
        )
        return InvestigationTrace(
            objective=objective,
            hypothesis=self.hypothesis,
            known_evidence=tuple(self.evidence),
            uncertainties=tuple(self.uncertainties),
            commands_run=tuple(self.commands_run),
            command_summaries=tuple(self.command_summaries),
            updated_understanding=understanding,
            next_investigation_step=(
                "Validate findings against retrieved RAG context and cross-check with "
                "model-driven analysis before high-confidence training use."
            ),
            confidence=confidence,
        )


class AgenticInvestigatorLoop:
    """Runs iterative CLI-assisted exploration for a single document topic.

    The loop mirrors the investigative workflow of a senior engineer working
    from a terminal:

    1. Identify relevant CLI commands for the topic.
    2. Execute each command against the repository.
    3. Parse output to extract file lists, symbol occurrences, or patterns.
    4. Accumulate findings into :class:`InvestigationMemory`.
    5. Refine the working hypothesis based on discovered evidence.
    6. Return a fully-populated :class:`InvestigationTrace`.
    """

    def __init__(self, cli: "CliExecutor") -> None:
        self.cli = cli

    def investigate(self, document: str, query: str, retrieval_confidence: float) -> InvestigationTrace:
        """Perform CLI exploration for *document* and return a structured trace."""
        memory = InvestigationMemory(document)
        objective = (
            f"Investigate '{document.removesuffix('.md').replace('_', ' ')}' "
            f"using CLI tooling seeded by query: {query!r}"
        )
        commands = _TOPIC_CLI_COMMANDS.get(document, _DEFAULT_CLI_COMMANDS)
        for args in commands:
            try:
                result = self.cli.run(args)
            except ValueError:
                continue
            summary = result.summary()
            memory.record_command(result, summary)
            if result.succeeded and result.stdout.strip():
                self._extract_evidence(memory, result)
        self._refine_hypothesis(memory, document)
        memory.updated_understanding = (
            f"CLI investigation of '{document}' executed {len(memory.commands_run)} command(s) "
            f"and gathered {len(memory.evidence)} evidence item(s). "
            "Findings are combined with vector-retrieved context for final document synthesis."
        )
        return memory.to_trace(objective, retrieval_confidence)

    @staticmethod
    def _extract_evidence(memory: InvestigationMemory, result: "CliResult") -> None:
        """Parse command output and add concrete evidence items to *memory*."""
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return
        # File-listing commands: record up to 5 distinct paths.
        if result.command.startswith("find "):
            paths = [ln for ln in lines if ln.startswith("./") or "/" in ln][:5]
            for p in paths:
                memory.add_evidence(f"Discovered file: {p}")
            if len(lines) > 5:
                memory.add_evidence(f"Additional {len(lines) - 5} files found by: {result.command}")
            return
        # Grep file-listing (-l flag): record matched files.
        if "-l" in result.command:
            for ln in lines[:5]:
                memory.add_evidence(f"Symbol match in file: {ln}")
            return
        # Grep line-number output: record first few matches.
        for ln in lines[:4]:
            memory.add_evidence(f"Pattern occurrence: {ln[:120]}")

    @staticmethod
    def _refine_hypothesis(memory: InvestigationMemory, document: str) -> None:
        """Update the working hypothesis based on accumulated evidence."""
        count = len(memory.evidence)
        if count == 0:
            memory.refine_hypothesis(
                f"No direct CLI evidence found for {document}; relying on RAG retrieval."
            )
        elif count <= 3:
            memory.refine_hypothesis(
                f"Sparse CLI evidence ({count} item(s)) suggests limited or indirect "
                f"{document.removesuffix('.md').replace('_', ' ')} patterns in this codebase."
            )
        else:
            memory.refine_hypothesis(
                f"CLI exploration yielded {count} evidence item(s), indicating "
                f"concrete {document.removesuffix('.md').replace('_', ' ')} artefacts are present and indexable."
            )


class InvestigatorAgent:
    """Produces the required finding documents from indexed evidence."""

    def __init__(
        self,
        paths: PipelinePaths,
        retriever: HybridRetriever,
        llm_client: "LLMClient | None" = None,
        cli_executor: "CliExecutor | None" = None,
    ) -> None:
        self.paths = paths
        self.retriever = retriever
        self.llm_client = llm_client
        self.cli_executor = cli_executor
        self._agentic_loop: AgenticInvestigatorLoop | None = (
            AgenticInvestigatorLoop(cli_executor) if cli_executor is not None else None
        )

    def run(self) -> dict[str, int]:
        self.paths.investigator_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for document in MANDATORY_DOCUMENTS:
            query = _DOCUMENT_QUERIES[document]
            hits = self.retriever.search(query, top_k=12)
            finding = self._finding_for(document, query, hits)
            model_analysis = self._model_analysis(document, hits)
            trace: InvestigationTrace | None = None
            if self._agentic_loop is not None:
                trace = self._agentic_loop.investigate(document, query, finding.confidence)
            target = self.paths.investigator_dir / document
            target.write_text(
                _render_document(document, query, finding, model_analysis, trace),
                encoding="utf-8",
            )
            written += 1
        return {"documents": written}

    def _model_analysis(self, document: str, hits: list[RetrievalHit]) -> str:
        """Return an LLM-generated analysis section, or a placeholder if no endpoint."""
        if self.llm_client is None:
            return "Configure an investigator endpoint to enable model-driven analysis."
        evidence_block = "\n".join(
            f"  [{i + 1}] {hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line}\n"
            f"      {hit.chunk.text[:300]}..."
            for i, hit in enumerate(hits[:4])
        )
        user_prompt = (
            f"Analyse the following retrieved evidence for the topic '{document}' "
            f"and produce a concise findings summary.\n\nEVIDENCE:\n{evidence_block}"
        )
        return self.llm_client.generate(_INVESTIGATOR_SYSTEM, user_prompt, max_tokens=512)

    @staticmethod
    def _finding_for(document: str, query: str, hits: list[RetrievalHit]) -> Finding:
        evidence = []
        refs = []
        for hit in hits[:5]:
            chunk = hit.chunk
            evidence.append(f"{chunk.path}:{chunk.start_line}-{chunk.end_line} score={hit.score:.3f}")
            refs.append(f"{chunk.path}:{chunk.start_line}-{chunk.end_line}")
        confidence = min(0.9, 0.25 + 0.08 * len(hits)) if hits else 0.15
        return Finding(
            title=document.removesuffix(".md").replace("_", " ").title(),
            category=document.removesuffix(".md"),
            confidence=confidence,
            supporting_evidence=tuple(evidence),
            source_file_references=tuple(refs),
            inferred_reasoning=(
                f"Evidence was retrieved with recursive investigation seed '{query}'. "
                "This automated pass records grounded leads for deeper heterogeneous-model analysis."
            ),
            unresolved_ambiguity=(
                "Automated static retrieval cannot prove runtime intent; model arbitration and source review "
                "should refine this finding before high-confidence training use."
            ),
        )


def _render_document(
    document: str,
    query: str,
    finding: Finding,
    model_analysis: str,
    trace: "InvestigationTrace | None" = None,
) -> str:
    trace_section = f"\n{trace.to_markdown()}\n" if trace is not None else ""
    return (
        f"# {document.removesuffix('.md').replace('_', ' ').title()}\n\n"
        "Generated by the Investigator Agent. All claims are intentionally evidence-scoped.\n\n"
        f"Investigation seed: `{query}`\n\n"
        f"{finding.to_markdown()}\n"
        f"{trace_section}"
        "## Model Analysis\n\n"
        f"{model_analysis}\n\n"
        "## Unresolved Questions\n"
        "- Which findings survive cross-model disagreement validation?\n"
        "- Which runtime paths require dynamic traces or test execution for confirmation?\n"
    )


def load_findings(directory: Path) -> list[dict[str, str]]:
    findings = []
    for path in sorted(directory.glob("*.md")):
        findings.append({"path": path.name, "text": path.read_text(encoding="utf-8")})
    return findings
