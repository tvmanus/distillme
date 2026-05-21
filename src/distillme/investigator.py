"""Investigator stage that emits evidence-backed architecture documents."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

from distillme.retrieval import RetrievalHit
from distillme.retrieval import HybridRetriever
from distillme.schemas import Finding, InvestigationIteration, InvestigationTrace, PipelinePaths

if TYPE_CHECKING:
    from distillme.cli_tools import CliExecutor, CliResult
    from distillme.inference import LLMClient

_INVESTIGATOR_SYSTEM = (
    "You are an expert software archaeologist performing deep codebase analysis. "
    "All claims must be grounded in the retrieved source evidence provided. "
    "Mark uncertainty explicitly. Do not fabricate class names, APIs, or runtime behavior."
)

# System prompt for the LLM-backed plan step in the two-step investigation loop.
_THINK_STEP_SYSTEM = (
    "You are a methodical codebase investigator planning the next CLI investigation step. "
    "Analyse the current evidence and decide exactly what to investigate next. "
    "Respond ONLY with a valid JSON object — no prose outside the JSON."
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

_SOURCE_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {".java", ".py", ".kt", ".go", ".ts", ".js", ".scala", ".cs", ".rb"}
)

# Default CLI commands used for topics not explicitly mapped above.
_DEFAULT_CLI_COMMANDS: list[list[str]] = [
    ["find", ".", "-name", "*.java", "-type", "f"],
    ["find", ".", "-name", "*.gradle", "-type", "f"],
]


@dataclasses.dataclass
class _InvestigationPlan:
    """Internal value-object produced by the planning step of one iteration."""

    rationale: str
    commands: list[list[str]]


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

    def to_trace(
        self,
        objective: str,
        confidence: float,
        iterations: list[InvestigationIteration] | None = None,
    ) -> InvestigationTrace:
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
            iterations=tuple(iterations) if iterations else (),
        )


class AgenticInvestigatorLoop:
    """Runs iterative CLI-assisted exploration for a single document topic.

    The loop implements a two-step **plan → execute** cycle repeated for up to
    *max_iterations* rounds:

    1. **Plan step** — the agent reviews evidence accumulated so far and
       decides which CLI commands to run next.  On the first iteration the
       static per-topic command list from :data:`_TOPIC_CLI_COMMANDS` is used
       as the seed.  On subsequent iterations the plan step performs
       *intelligent branching*: when an LLM client is provided it reasons over
       the accumulated evidence and returns a structured list of targeted
       follow-up commands; when no LLM is configured,
       :meth:`_derive_followup_commands` applies a heuristic strategy —
       inspecting symbols in discovered source files, listing sibling
       directories, or examining git history.

    2. **Execute step** — each planned command is run via the
       :class:`~distillme.cli_tools.CliExecutor`; output is parsed into
       evidence items and appended to :class:`InvestigationMemory`.

    The working hypothesis is refined after every execute step.  The loop
    terminates early when an execute step yields no new evidence, avoiding
    wasted commands.

    Each iteration is recorded as an :class:`~distillme.schemas.InvestigationIteration`
    and attached to the returned :class:`~distillme.schemas.InvestigationTrace`.
    """

    def __init__(
        self,
        cli: "CliExecutor",
        max_iterations: int = 3,
        llm_client: "LLMClient | None" = None,
    ) -> None:
        self.cli = cli
        self.max_iterations = max_iterations
        self._llm_client = llm_client

    def investigate(self, document: str, query: str, retrieval_confidence: float) -> InvestigationTrace:
        """Perform multi-iteration CLI exploration for *document* and return a structured trace."""
        memory = InvestigationMemory(document)
        objective = (
            f"Investigate '{document.removesuffix('.md').replace('_', ' ')}' "
            f"using CLI tooling seeded by query: {query!r}"
        )
        recorded_iterations: list[InvestigationIteration] = []

        for iteration in range(self.max_iterations):
            # ── Step 1: PLAN ────────────────────────────────────────────────
            plan = self._plan_step(memory, document, iteration)

            # ── Step 2: EXECUTE ─────────────────────────────────────────────
            evidence_before = len(memory.evidence)
            for args in plan.commands:
                try:
                    result = self.cli.run(args)
                except ValueError:
                    continue
                summary = result.summary()
                memory.record_command(result, summary)
                if result.succeeded and result.stdout.strip():
                    self._extract_evidence(memory, result)

            new_findings = tuple(memory.evidence[evidence_before:])
            self._refine_hypothesis(memory, document)

            recorded_iterations.append(
                InvestigationIteration(
                    iteration_num=iteration,
                    plan_rationale=plan.rationale,
                    commands_planned=tuple(" ".join(c) for c in plan.commands),
                    findings=new_findings,
                    hypothesis_after=memory.hypothesis,
                )
            )

            # Early-stop: no new findings after the first seed iteration.
            if iteration > 0 and not new_findings:
                break

        memory.updated_understanding = (
            f"Multi-iteration CLI investigation of '{document}' completed "
            f"{len(recorded_iterations)} iteration(s), executing {len(memory.commands_run)} "
            f"command(s) and gathering {len(memory.evidence)} evidence item(s). "
            "Findings are combined with vector-retrieved context for final document synthesis."
        )
        return memory.to_trace(objective, retrieval_confidence, recorded_iterations)

    def _plan_step(
        self, memory: InvestigationMemory, document: str, iteration: int
    ) -> _InvestigationPlan:
        """Return the next batch of commands to execute.

        Iteration 0 uses the static per-topic seed commands.  Later iterations
        branch intelligently: when an LLM client is configured
        :meth:`_plan_step_llm` is attempted first; on failure (or when no LLM
        is available) the heuristic :meth:`_derive_followup_commands` is used.
        """
        if iteration == 0:
            cmds = list(_TOPIC_CLI_COMMANDS.get(document, _DEFAULT_CLI_COMMANDS))
            return _InvestigationPlan(
                rationale=(
                    f"Initial broad exploration of "
                    f"'{document.removesuffix('.md').replace('_', ' ')}'"
                ),
                commands=cmds,
            )
        if self._llm_client is not None:
            return self._plan_step_llm(memory, document, iteration)
        return self._derive_followup_commands(memory, document, iteration)

    def _plan_step_llm(
        self, memory: InvestigationMemory, document: str, iteration: int
    ) -> _InvestigationPlan:
        """Use the LLM to reason about the next investigation step.

        Builds a structured prompt from the current evidence state, requests a
        JSON response containing a rationale and a list of CLI commands, then
        validates the output before constructing an :class:`_InvestigationPlan`.
        Falls back to :meth:`_derive_followup_commands` when the LLM returns
        unparseable or invalid output.
        """
        topic = document.removesuffix(".md").replace("_", " ")
        evidence_str = (
            "\n".join(f"  - {e}" for e in memory.evidence[-8:]) or "  - None gathered"
        )
        commands_str = (
            "\n".join(f"  - {c}" for c in memory.commands_run[-5:]) or "  - None"
        )
        user_prompt = (
            f"Topic: {topic}\n"
            f"Iteration: {iteration + 1}\n"
            f"Current hypothesis: {memory.hypothesis}\n"
            f"Evidence gathered ({len(memory.evidence)} items):\n{evidence_str}\n"
            f"Commands already run:\n{commands_str}\n\n"
            "Decide the next investigation step. Respond with JSON only:\n"
            "{\n"
            '  "rationale": "<1-2 sentences on what to investigate next>",\n'
            '  "commands": [["grep", "-En", "class|interface", "./path/File.java"]]\n'
            "}\n\n"
            "Rules:\n"
            "- commands: 0-4 entries using only: find, grep, git, ls, cat, head, wc\n"
            "- Target specific files or patterns discovered in the evidence above\n"
            "- If investigation is complete, use [] for commands"
        )
        try:
            raw = self._llm_client.generate(_THINK_STEP_SYSTEM, user_prompt, max_tokens=512)
            text = raw.strip()
            # Strip optional markdown code fence.
            if text.startswith("```"):
                parts = text.split("```", 2)
                text = parts[1][4:] if parts[1].startswith("json") else parts[1]
            data = json.loads(text)
            rationale = str(data.get("rationale", "")).strip()
            raw_cmds = data.get("commands", [])
            validated: list[list[str]] = []
            for cmd in raw_cmds[:4]:
                if isinstance(cmd, list) and cmd:
                    validated.append([str(a) for a in cmd])
            if rationale and validated:
                return _InvestigationPlan(rationale=f"[LLM] {rationale}", commands=validated)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        # LLM returned unparseable or empty output; use heuristic fallback.
        return self._derive_followup_commands(memory, document, iteration)

    @staticmethod
    def _derive_followup_commands(
        memory: InvestigationMemory, document: str, iteration: int
    ) -> _InvestigationPlan:
        """Generate follow-up commands by branching from accumulated evidence.

        Strategy:
        - Discovered source files → inspect their symbol structure via grep.
        - Discovered file paths    → list the parent directory.
        - Iteration ≥ 2 with no other targets → sample git log for change context.
        - No targets at all       → repeat the base exploration commands.
        """
        commands: list[list[str]] = []
        rationale_parts: list[str] = []

        # ── Collect discovered file paths from evidence ────────────────────
        discovered_files: list[str] = []
        for ev in memory.evidence:
            for prefix in ("Discovered file: ", "Symbol match in file: "):
                if ev.startswith(prefix):
                    path = ev.removeprefix(prefix).strip()
                    if path not in discovered_files:
                        discovered_files.append(path)

        # ── Branch 1: inspect symbol structure of discovered source files ──
        inspected = 0
        for filepath in discovered_files:
            suffix = "." + filepath.rsplit(".", 1)[-1] if "." in filepath else ""
            if suffix in _SOURCE_FILE_EXTENSIONS and inspected < 3:
                commands.append(
                    ["grep", "-En", "class|interface|enum|def |func ", filepath]
                )
                rationale_parts.append(f"Inspect symbols in {filepath}")
                inspected += 1

        # ── Branch 2: list parent directories of discovered files ──────────
        parent_dirs: set[str] = set()
        for filepath in discovered_files:
            normalized = filepath.lstrip("./")
            parts = normalized.split("/")
            if len(parts) > 1:
                parent_dirs.add("./" + "/".join(parts[:-1]))
        for pkg_dir in sorted(parent_dirs)[:2]:
            commands.append(["ls", pkg_dir])
            rationale_parts.append(f"List directory {pkg_dir}")

        # ── Branch 3: git log for temporal context on deeper iterations ────
        if iteration >= 2 and not commands:
            commands.append(["git", "log", "--oneline", "-10"])
            rationale_parts.append("Sample git history for change context")

        # ── Fallback: repeat base exploration if no branch targets found ───
        if not commands:
            commands = list(_TOPIC_CLI_COMMANDS.get(document, _DEFAULT_CLI_COMMANDS))
            rationale_parts.append("No branch targets found; repeating base exploration")

        return _InvestigationPlan(
            rationale="; ".join(rationale_parts),
            commands=commands,
        )

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
                f"concrete {document.removesuffix('.md').replace('_', ' ')} artifacts are present and indexable."
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
            AgenticInvestigatorLoop(cli_executor, llm_client=llm_client)
            if cli_executor is not None else None
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
