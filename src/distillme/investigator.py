"""Investigator stage that emits evidence-backed architecture documents."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

_LOG = logging.getLogger(__name__)


def _extract_xml_block(text: str, tag: str) -> str:
    """Return the content of the first ``<tag>…</tag>`` block in *text*, stripped.

    Returns an empty string when the tag is absent.
    """
    pattern = re.compile(rf"<{re.escape(tag)}>([ \S]*?)</{re.escape(tag)}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


from distillme.retrieval import RetrievalHit
from distillme.retrieval import HybridRetriever
from distillme.schemas import Finding, InvestigationIteration, InvestigationTrace, PipelinePaths

if TYPE_CHECKING:
    from distillme.cli_tools import CliExecutor, CliResult
    from distillme.inference import LLMClient

_INVESTIGATOR_SYSTEM = """\
You are a senior software architect performing forensic codebase analysis for knowledge distillation.
Your analysis will be used to train AI coding assistants — accuracy, depth, and specificity matter enormously.

INVESTIGATION MANDATE:
- Go deep. Surface-level descriptions are insufficient. Explain the WHY behind every design decision you observe.
- Read code directly. Your analysis must be grounded in actual source you have seen, not inferred from names.
- Name specifics. Reference exact class names, method signatures, package paths, and line numbers.
- Identify patterns. Recognise design patterns, algorithms, architectural styles, and idioms in use.
- Explain interactions. Describe how components collaborate and communicate, not just what each does alone.
- Track purpose. For each component, answer: what problem does it solve? Why was it designed this way?

EVIDENCE DISCIPLINE:
- Every non-trivial claim must reference a specific class, method, or file location you have seen.
- Distinguish between what you observe directly in code versus what you infer from naming or structure.
- If you cannot confirm something from the evidence provided, say so explicitly.
- Do not fabricate class names, method signatures, API contracts, or runtime behaviours.

OUTPUT QUALITY BAR:
- Write as if explaining this codebase to a senior engineer joining the team on day one.
- A good analysis names 10+ specific classes/methods/algorithms and explains their relationships.
- Prefer precise technical language over vague generalities.
- Identify architectural trade-offs and design decisions, not just component inventories.
"""

# System prompt for the global reconnaissance synthesis (runs once per pipeline run).
_RECON_SYSTEM = """\
You are a senior software architect performing rapid codebase reconnaissance.
Produce a concise, factual map of this codebase's module structure in 2-3 paragraphs.
Focus on: top-level module breakdown, apparent purpose of each subsystem, programming language
and framework patterns, and inter-module relationships.
Be concrete — name the actual module directories and packages you see. Do not speculate.
This summary will be used as shared orientation context for 22 separate topic investigations.
"""

# System prompt for the LLM-backed plan step in the iterative investigation loop.
_THINK_STEP_SYSTEM = (
    "You are a methodical codebase investigator deciding which source files to read next. "
    "You have the following CLI tools available:\n"
    "  head -N FILE          Read the first N lines of a file (use for full file overview).\n"
    "  grep_context PAT TARGET [N]  Find pattern (extended regex) in file or directory and return\n"
    "                        N lines of code around each match (default 40).  Use | for alternation.\n"
    "                        PREFER THIS over head when: tracing inheritance/interfaces, finding a\n"
    "                        specific method across files, or following a symbol scattered across\n"
    "                        parent/base classes.  Keeps context small and targeted.\n"
    "  grep -rn PAT DIR      Find files containing a pattern (use to discover relevant files).\n"
    "  find DIR -name PAT    Discover files by name.\n"
    "  git log/blame         Understand change history.\n"
    "\nMulti-step tracing strategy:\n"
    "  1. Use grep/find to discover which files contain a symbol.\n"
    "  2. Use grep_context to extract the exact definition/usage block.\n"
    "  3. Use grep_context to follow the inheritance chain upward.\n"
    "  4. Use head only when you need the full file structure.\n"
    "\nRespond ONLY with a valid JSON object — no prose outside the JSON."
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

# Per-topic synthesis questions used to guide the LLM when it analyses collected source code.
# These replace the generic "summarise this evidence" prompt with targeted engineering questions.
_TOPIC_SYNTHESIS_QUESTIONS: dict[str, str] = {
    "architecture_overview.md": (
        "What is the high-level architecture of this codebase? Describe the main modules or "
        "subsystems, their responsibilities, and how they interact. What architectural patterns "
        "are used (layered, plugin-based, event-driven, etc.)?"
    ),
    "service_catalog.md": (
        "What services, controllers, and components exist? For each, describe its responsibility, "
        "the domain it serves, and the interfaces it exposes."
    ),
    "package_relationships.md": (
        "Describe the package hierarchy and inter-package dependencies. Which packages are "
        "foundational vs. domain-specific? Do any dependency cycles or layering violations exist?"
    ),
    "domain_model.md": (
        "What are the core domain entities, aggregates, and value objects? What business rules "
        "or invariants are encoded in them, and how are they related to each other?"
    ),
    "coding_conventions.md": (
        "What naming conventions, code organisation patterns, error handling styles, "
        "and API design conventions are consistently applied across this codebase?"
    ),
    "testing_strategy.md": (
        "What testing approaches are used (unit, integration, end-to-end)? What test frameworks "
        "and mocking libraries are present? Which areas appear under-tested?"
    ),
    "algorithms_catalog.md": (
        "What specific algorithms, data structures, and computational patterns are implemented? "
        "For each, describe its purpose, where it is used, its inputs/outputs, and any notable "
        "complexity or optimisation. Do not list file paths — describe the actual logic."
    ),
    "concurrency_patterns.md": (
        "What concurrency and parallelism patterns are used? Where are threads, executors, locks, "
        "reactive streams, or async constructs used, and what problems do they solve?"
    ),
    "security_analysis.md": (
        "What security controls are implemented (authentication, authorisation, input validation, "
        "secret management, cryptography)? Where are potential vulnerabilities or missing controls?"
    ),
    "dependency_analysis.md": (
        "What are the key external dependencies? Why is each dependency used and what does it "
        "provide? Are there any outdated, duplicated, or risky dependencies?"
    ),
    "api_behavior.md": (
        "What APIs does this codebase expose? Describe the endpoints, request/response contracts, "
        "error handling behaviour, and any versioning or rate-limiting strategies."
    ),
    "database_model.md": (
        "What is the data model? Describe the entities stored, the schema design, repository "
        "patterns, transaction boundaries, and any migration strategy."
    ),
    "execution_flows.md": (
        "What are the main execution flows? Starting from the entry points, trace the key "
        "request-handling or job-processing paths end-to-end."
    ),
    "anti_patterns.md": (
        "What anti-patterns, code smells, or technical debt items exist? "
        "Cite specific locations and explain the risk or impact of each."
    ),
    "performance_characteristics.md": (
        "What are the performance-critical paths? Where are caches, batching, lazy evaluation, "
        "or other optimisations applied, and what bottlenecks remain?"
    ),
    "operational_behavior.md": (
        "How is this system operated? What logging, metrics, health checks, configuration "
        "mechanisms, and deployment artefacts are present?"
    ),
    "configuration_model.md": (
        "How is the system configured? What properties exist, how are they loaded, what are "
        "the defaults, and which can be changed at runtime?"
    ),
    "event_flow_analysis.md": (
        "What event-driven patterns are used? Describe the event types, producers, consumers, "
        "and the flow of events through the system."
    ),
    "state_machine_analysis.md": (
        "What state machines or lifecycle models exist? For each, describe the states, "
        "transitions, triggers, and the domain concept being modelled."
    ),
    "caching_behavior.md": (
        "What caching strategies are employed? Describe the cache scopes (in-process, "
        "distributed), eviction policies, TTLs, invalidation strategies, and what is cached."
    ),
    "exception_taxonomy.md": (
        "What exception hierarchy exists? How are exceptions classified (recoverable vs. fatal, "
        "domain vs. infrastructure)? What retry or fallback strategies are used?"
    ),
    "glossary.md": (
        "Define the domain-specific terminology used in this codebase. For each term, provide "
        "the definition as it is used here, not a generic textbook definition."
    ),
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

# ---------------------------------------------------------------------------
# Token budget constants — calibrated for models with up to 256k context.
# Reduce if using smaller models with tighter context windows.
# ---------------------------------------------------------------------------
# ── Input limits ────────────────────────────────────────────────────────────
# Hardware: 2× RTX 5090, 64 GB VRAM, model fully offloaded → ~100 tok/s.
# Context window: 250 000 tokens (≈875 000 chars at ~3.5 ch/tok).
# With 20 files × 32 k chars + 24 RAG chunks × 3 k chars the total input
# sits at ~208 k tokens, comfortably inside the 250 k window.
_FILE_LINES_PER_READ: int = 500       # head -500 covers most Java files in full (~20k chars)
_FILE_CHARS_PER_READ: int = 32_000    # Hard char cap per file (~500 lines of Java)
_MAX_FILES_PER_SYNTHESIS: int = 20    # 20 files × 32k chars = 640k chars input
_RAG_CHUNKS_PER_TOPIC: int = 24       # Wider semantic coverage per topic
_RAG_CHUNK_CHARS: int = 3_000         # Richer context per RAG snippet

# ── Output budgets ───────────────────────────────────────────────────────────
# Qwen3 extended-thinking burns CoT tokens before the answer.
# Typical CoT: 2 000–5 000 tokens.  Budgets give ample room for both.
# At 100 tok/s wall-clock: synthesis ≤ 2.7 min, analysis ≤ 5.3 min.
# "Critical" calls (synthesis, analysis) are intentionally uncapped relative
# to answer quality — let the model go as deep as the topic warrants.
_SYNTHESIS_MAX_TOKENS: int = 16_000   # ~4k CoT + up to 12k thorough analysis
_ANALYSIS_MAX_TOKENS: int = 32_000    # ~6k CoT + up to 26k exhaustive final analysis
_PLAN_MAX_TOKENS: int = 8_000         # ~3k CoT + careful multi-command plan
_RECON_MAX_TOKENS: int = 12_000       # ~4k CoT + full codebase orientation


@dataclasses.dataclass
class _CodebaseContext:
    """Global codebase reconnaissance shared across all 22 topic investigations.

    Built once at the start of the investigate stage and injected into every
    :class:`AgenticInvestigatorLoop` call and :meth:`InvestigatorAgent._model_analysis`
    so each topic investigation starts with a complete picture of the codebase shape.
    """

    directory_tree: str       # Compressed find -type d output
    module_overview: str      # LLM-generated 2-3 paragraph structural overview
    total_source_files: int   # Number of indexable source files found
    build_files_summary: str  # Concatenated content of key build/manifest files


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
        self.file_contents: list[tuple[str, str]] = []  # [(filepath, content)] buffered for LLM synthesis
        self.synthesized_insights: list[str] = []  # LLM-produced insights from file content

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
        max_iterations: int = 5,
        llm_client: "LLMClient | None" = None,
        codebase_context: "_CodebaseContext | None" = None,
    ) -> None:
        self.cli = cli
        self.max_iterations = max_iterations
        self._llm_client = llm_client
        self.codebase_context = codebase_context

    def investigate(
        self,
        document: str,
        query: str,
        retrieval_confidence: float,
        rag_hits: "list[RetrievalHit] | None" = None,
    ) -> tuple[InvestigationTrace, list[str]]:
        """Perform multi-iteration CLI and LLM exploration for *document*.

        On iteration 0, :meth:`_plan_step` primes the read list with RAG-retrieved files
        most semantically similar to the topic so the LLM immediately reads relevant source
        rather than doing generic discovery.  Subsequent iterations use the LLM planner or
        heuristic fallback.

        Returns the :class:`InvestigationTrace` and the list of LLM-synthesised insights.
        """
        memory = InvestigationMemory(document)
        objective = (
            f"Investigate '{document.removesuffix('.md').replace('_', ' ')}' "
            f"using CLI tooling seeded by query: {query!r}"
        )
        recorded_iterations: list[InvestigationIteration] = []
        seen_command_sets: set[frozenset[tuple[str, ...]]] = set()

        for iteration in range(self.max_iterations):
            # ── Step 1: PLAN ────────────────────────────────────────────────
            plan = self._plan_step(memory, document, iteration, rag_hits=rag_hits)

            # Dedup guard: stop immediately if we would repeat the same command set.
            cmd_key: frozenset[tuple[str, ...]] = frozenset(tuple(c) for c in plan.commands)
            if iteration > 0 and cmd_key in seen_command_sets:
                break
            seen_command_sets.add(cmd_key)

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

            # ── Step 3: SYNTHESISE ───────────────────────────────────────────
            # Ask the LLM to analyse any source file content buffered during execution.
            # This is the principal answer; CLI output is only the navigation layer.
            insight = self._synthesize_iteration_insight(memory, document)
            if insight:
                memory.synthesized_insights.append(insight)

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

            # Early-stop: no new evidence and no new insight after the seed iteration.
            if iteration > 0 and not new_findings and not insight:
                break

        memory.updated_understanding = (
            f"Multi-iteration CLI investigation of '{document}' completed "
            f"{len(recorded_iterations)} iteration(s), executing {len(memory.commands_run)} "
            f"command(s), gathering {len(memory.evidence)} evidence item(s), and producing "
            f"{len(memory.synthesized_insights)} synthesised insight(s). "
            "Findings are combined with vector-retrieved context for final document synthesis."
        )
        return memory.to_trace(objective, retrieval_confidence, recorded_iterations), memory.synthesized_insights

    def _plan_step(
        self,
        memory: InvestigationMemory,
        document: str,
        iteration: int,
        rag_hits: "list[RetrievalHit] | None" = None,
    ) -> _InvestigationPlan:
        """Return the next batch of commands to execute.

        Iteration 0 reads RAG-retrieved files first (direct semantic match to the topic),
        then appends topic-specific CLI discovery commands.  Later iterations use the LLM
        planner when available or the heuristic fallback.
        """
        if iteration == 0:
            commands: list[list[str]] = []
            rationale_parts: list[str] = []

            # PRIMARY: read the files the RAG index judged most relevant.
            # Direct file reads give the LLM actual source code immediately.
            seen_paths: set[str] = set()
            if rag_hits:
                for hit in rag_hits[:_MAX_FILES_PER_SYNTHESIS]:
                    path = hit.chunk.path
                    if path not in seen_paths:
                        seen_paths.add(path)
                        commands.append(["head", f"-{_FILE_LINES_PER_READ}", path])
                        rationale_parts.append(
                            f"RAG-retrieved {path[:70]} (score={hit.score:.2f})"
                        )

            # SECONDARY: topic-specific CLI exploration to catch files missed by RAG.
            commands.extend(_TOPIC_CLI_COMMANDS.get(document, _DEFAULT_CLI_COMMANDS))
            rationale_parts.append(f"CLI seed for '{document}'")

            return _InvestigationPlan(
                rationale="; ".join(rationale_parts),
                commands=commands,
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
            "\n".join(f"  - {e}" for e in memory.evidence[-10:]) or "  - None gathered"
        )
        commands_str = (
            "\n".join(f"  - {c}" for c in memory.commands_run[-8:]) or "  - None"
        )
        insights_str = (
            "\n\n---\n\n".join(memory.synthesized_insights[-2:])
            or "None yet — deeper reading is needed."
        )
        codebase_ctx = ""
        if self.codebase_context:
            codebase_ctx = (
                f"CODEBASE CONTEXT:\n{self.codebase_context.module_overview}\n\n"
            )
        synthesis_question = _TOPIC_SYNTHESIS_QUESTIONS.get(
            document,
            f"What are the key insights about {topic} in this codebase?",
        )
        user_prompt = (
            f"{codebase_ctx}"
            f"TOPIC: {topic}\n"
            f"INVESTIGATION QUESTION: {synthesis_question}\n\n"
            f"ITERATION: {iteration + 1} of {self.max_iterations}\n"
            f"SYNTHESISED INSIGHTS SO FAR:\n{insights_str}\n\n"
            f"EVIDENCE GATHERED ({len(memory.evidence)} items):\n{evidence_str}\n\n"
            f"COMMANDS ALREADY RUN:\n{commands_str}\n\n"
            "AVAILABLE TOOLS:\n"
            f'  ["head", "-{_FILE_LINES_PER_READ}", "./path/File.java"]\n'
            f'     → Read first {_FILE_LINES_PER_READ} lines of a file (full overview).\n'
            f'  ["grep_context", "ClassName|IFaceName", ".", "60"]\n'
            f'     → Find pattern (extended regex, | for OR) anywhere in repo, return 60-line\n'
            f'       code blocks around each match.  USE THIS to:\n'
            f'         • Locate a class/interface definition you know the name of\n'
            f'         • Trace where an interface is implemented (grep_context "implements IFace" . 60)\n'
            f'         • Follow inheritance up to parent classes (grep_context "class Parent" . 60)\n'
            f'         • Find all callers/overrides of a method across many files\n'
            f'         • Get focused logic without reading entire files\n'
            f'  ["grep", "-rn", "--include=*.java", "pattern", "."]\n'
            f'     → Discover file paths matching a pattern.\n'
            f'  ["find", ".", "-name", "*.java", "-type", "f"]\n'
            f'     → List all source files.\n\n'
            "TASK: Plan the next investigation step to better answer the question.\n"
            "Use grep_context for focused symbol tracing. Use head only for full-file reads.\n"
            "Respond with JSON only:\n"
            "{\n"
            '  "rationale": "<1-2 sentences: what gap are you filling and why these tools?>",\n'
            f'  "commands": [["grep_context", "class BaseClass|interface IFace", ".", "60"]]\n'
            "}\n\n"
            "RULES:\n"
            f"- Up to {_MAX_FILES_PER_SYNTHESIS} commands total\n"
            "- Use only: grep_context, head, grep, find, git, ls, wc\n"
            "- grep_context patterns use extended regex: | for OR, .* for wildcard\n"
            "- Target symbols/files not yet examined that are likely to answer the question\n"
            "- For inheritance chains: first grep_context the interface, then grep_context its implementors\n"
            "- If the question is fully answered by existing insights, use [] for commands"
        )
        try:
            raw = self._llm_client.generate(_THINK_STEP_SYSTEM, user_prompt, max_tokens=_PLAN_MAX_TOKENS)
            text = raw.strip()
            # Strip optional markdown code fence.
            if text.startswith("```"):
                parts = text.split("```", 2)
                body = parts[1] if len(parts) > 1 else ""
                text = body[4:] if body.startswith("json") else body
            data = json.loads(text)
            rationale = str(data.get("rationale", "")).strip()
            raw_cmds = data.get("commands", [])
            validated: list[list[str]] = []
            for cmd in raw_cmds[:_MAX_FILES_PER_SYNTHESIS]:
                if isinstance(cmd, list) and cmd:
                    validated.append([str(a) for a in cmd])
            if rationale and validated:
                return _InvestigationPlan(rationale=f"[LLM] {rationale}", commands=validated)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            _LOG.warning(
                "LLM plan step returned unparseable output (topic=%r, iter=%d): %s",
                document, iteration, exc,
            )
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

        # ── Branch 1: read content of discovered source files ──────────────
        # head is used so the LLM sees actual code logic, not just symbol names.
        inspected = 0
        for filepath in discovered_files:
            suffix = "." + filepath.rsplit(".", 1)[-1] if "." in filepath else ""
            if suffix in _SOURCE_FILE_EXTENSIONS and inspected < _MAX_FILES_PER_SYNTHESIS:
                commands.append(["head", f"-{_FILE_LINES_PER_READ}", filepath])
                rationale_parts.append(f"Read content of {filepath}")
                inspected += 1

        # ── Branch 2: trace inheritance / interfaces seen in evidence ──────
        # When prior grep output captured `extends X` or `implements Y`,
        # automatically follow the symbol to its definition — the same
        # strategy used by SOTA agents (SWE-agent, Aider) to traverse
        # class hierarchies without reading entire file trees.
        traced_symbols: set[str] = set()
        for ev in memory.evidence:
            text = ev.removeprefix("Pattern occurrence:").strip()
            for keyword in ("extends ", "implements ", "class "):
                if keyword in text:
                    after = text.split(keyword, 1)[-1].strip()
                    name = after.split()[0].rstrip("{,<(").strip() if after.split() else ""
                    if (
                        name
                        and len(name) > 3
                        and name[0].isupper()
                        and name not in traced_symbols
                        and len(commands) < _MAX_FILES_PER_SYNTHESIS
                    ):
                        commands.append(
                            ["grep_context", f"class {name}|interface {name}", ".", "60"]
                        )
                        traced_symbols.add(name)
                        rationale_parts.append(f"Trace definition of {name}")
                    break

        # ── Branch 3: list parent directories of discovered files ──────────
        parent_dirs: set[str] = set()
        for filepath in discovered_files:
            normalized = filepath.lstrip("./")
            parts = normalized.split("/")
            if len(parts) > 1:
                parent_dirs.add("./" + "/".join(parts[:-1]))
        for pkg_dir in sorted(parent_dirs)[:3]:
            commands.append(["ls", pkg_dir])
            rationale_parts.append(f"List directory {pkg_dir}")

        # ── Branch 4: git log for temporal context on deeper iterations ────
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
        cmd_first = result.command.split()[0]

        # ── grep_context: focused code blocks ────────────────────────────────────
        # Output is pre-formatted with === file:lineno === headers; buffer as a
        # virtual file so the synthesis LLM sees precisely the relevant blocks.
        if cmd_first == "grep_context":
            content = result.stdout[:_FILE_CHARS_PER_READ]
            if content.strip():
                # Label shows the search pattern for context in the synthesis prompt.
                parts = result.command.split(None, 3)
                label = f"[grep_context: {parts[1]!r} in {parts[2] if len(parts) > 2 else '.'!r}]"
                memory.file_contents.append((label, content))
                # Also record each matched file as a discovered evidence item.
                for ln in result.stdout.splitlines():
                    if ln.startswith("===") and ":" in ln:
                        loc = ln.strip("= ").split(":")[0]
                        if loc:
                            memory.add_evidence(f"Focused extract from: {loc}")
            return

        # ── File content commands (head / cat) ──────────────────────────────────
        if cmd_first in ("head", "cat"):
            filepath = result.command.split()[-1]
            content = result.stdout[:_FILE_CHARS_PER_READ]
            if content.strip():
                memory.file_contents.append((filepath, content))
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

    def _synthesize_iteration_insight(
        self,
        memory: InvestigationMemory,
        document: str,
    ) -> str | None:
        """Use the LLM to analyse buffered file content and return a substantive insight.

        This is the principal source of analytical value in the investigation loop.
        CLI commands surface relevant files; this method reads them and asks the LLM
        the topic-specific question from :data:`_TOPIC_SYNTHESIS_QUESTIONS`.
        Consumes and clears :attr:`InvestigationMemory.file_contents`.
        Returns ``None`` when no LLM client is configured or no content was buffered.
        """
        if self._llm_client is None or not memory.file_contents:
            return None
        topic = document.removesuffix(".md").replace("_", " ")
        synthesis_question = _TOPIC_SYNTHESIS_QUESTIONS.get(
            document,
            f"What is the purpose and logic of this code in relation to {topic}? "
            "Name specific classes, methods, and patterns.",
        )
        # Global codebase orientation keeps the LLM anchored to the overall structure.
        context_header = ""
        if self.codebase_context:
            context_header = (
                f"CODEBASE CONTEXT (for orientation):\n{self.codebase_context.module_overview}\n\n"
            )
        # Prior insights let the LLM build on previous iterations rather than repeating.
        prior_block = ""
        if memory.synthesized_insights:
            prior_block = (
                "PREVIOUS ITERATION FINDINGS (build on these, avoid repetition):\n"
                + "\n---\n".join(memory.synthesized_insights[-2:])
                + "\n\n"
            )
        content_block = "\n".join(
            f"=== {fp} ===\n{content}"
            for fp, content in memory.file_contents[:_MAX_FILES_PER_SYNTHESIS]
        )
        user_prompt = (
            f"{context_header}"
            f"TOPIC: {topic}\n"
            f"QUESTION: {synthesis_question}\n\n"
            f"{prior_block}"
            f"SOURCE CODE TO ANALYSE:\n{content_block}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read ALL source code above carefully before writing.\n"
            "2. Answer the question with code-grounded, specific analysis.\n"
            "3. Name exact classes, methods, algorithms, and design patterns.\n"
            "4. Explain the PURPOSE of what you find — why does this code exist?\n"
            "5. Identify relationships: how do the classes you see collaborate?\n"
            "6. If code does not directly address the topic, describe its actual purpose.\n"
            "7. Write as if explaining to a senior engineer joining the team.\n"
            "Write 400-700 words."
        )
        try:
            result = self._llm_client.generate(_INVESTIGATOR_SYSTEM, user_prompt, max_tokens=_SYNTHESIS_MAX_TOKENS)
            memory.file_contents.clear()
            if not result or not result.strip():
                _LOG.warning("LLM synthesis returned empty response for topic %r", document)
                return None
            return result.strip()
        except Exception as exc:
            _LOG.exception(
                "LLM synthesis failed for topic %r (%d file(s), prompt ~%d chars): %s",
                document, len(memory.file_contents), len(user_prompt), exc,
            )
            memory.file_contents.clear()
            return None

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


# ---------------------------------------------------------------------------
# Multi-turn tool-use loop for evidence-grounded investigator synthesis
# ---------------------------------------------------------------------------


class _InvestigatorMultiTurnLoop:
    """Multi-turn, tool-equipped loop for evidence-grounded investigator document synthesis.

    Architecture
    ------------
    Replaces :class:`AgenticInvestigatorLoop`'s plan → execute → synthesize triple
    per iteration with a unified XML-protocol conversation loop.  Each turn:

    ``<think>``   Private scratchpad — STRIPPED after each turn.  The LLM
                  reasons freely about what to explore next without that
                  reasoning polluting the context window.

    ``<tools>``   Up to ``MAX_TOOL_CALLS_PER_TURN`` JSON objects (one per line).
                  Results are injected into the *next* prompt only, then discarded;
                  the LLM must distil findings into ``<memo>`` to retain them.

    ``<memo>``    Compressed architectural findings carried forward.  Accumulated
                  across all turns up to ``MAX_MEMO_TOTAL_CHARS``; oldest entries
                  are evicted when the cap is exceeded.

    ``<final>``   Complete markdown analysis document.  Terminates the loop
                  immediately when a non-empty block is found.

    Memo budget rationale (``MAX_MEMO_TOTAL_CHARS = 16_000``)
    ---------------------------------------------------------
    ``_SYNTHESIS_MAX_TOKENS = 16_000`` is the token budget for one per-iteration
    synthesis call.  The memo stores *compressed* versions of those findings,
    targeting ~2 kB per turn over 8 turns.  At 16 kB memo + 20 kB tool results
    (5 calls × 4 kB) + ~5 kB system/headers, the total continuation prompt stays
    under ~41 kB — well inside the 875 kB context window.  This is 2.67× the
    teacher synthesizer's 6 kB limit, proportional to the investigator's longer
    output documents (500+ word architectural analyses vs. 12 Q&A pairs).
    """

    MAX_TOOL_CALLS_PER_TURN: int = 5
    MAX_TOOL_OUTPUT_CHARS: int = 4_000
    MAX_MEMO_TOTAL_CHARS: int = 16_000   # = _SYNTHESIS_MAX_TOKENS chars; 2.67× teacher

    _TOOL_PROTOCOL: str = (
        "\n\n## Multi-Turn Investigation Protocol\n"
        "Use EXACTLY these XML tags every turn:\n\n"
        "<think>\n"
        "Private scratchpad — what evidence is missing, which files to read next.\n"
        "STRIPPED from all subsequent prompts. Write freely.\n"
        "</think>\n\n"
        "<tools>\n"
        '{"tool": "search", "query": "ClusterDriver connection", "top_k": 8}\n'
        '{"tool": "grep_context", "pattern": "implements.*Reconnectable", "path": ".", "context_lines": 60}\n'
        "</tools>\n\n"
        "<memo>\n"
        "Compressed findings — cite EXACT class names, method names, file paths.\n"
        "PERSISTS across all subsequent turns. Be dense and specific. Omit if nothing new.\n"
        "</memo>\n\n"
        "When you have sufficient evidence for a thorough analysis, emit:\n\n"
        "<final>\n"
        "## Executive Summary\n...\n\n"
        "## Detailed Findings\n...\n\n"
        "## Design Rationale\n...\n\n"
        "## Key Architectural Observations\n...\n"
        "</final>\n\n"
        "## Available Tools\n"
        "  search         Semantic chunk search.\n"
        '                 {"tool": "search", "query": "...", "top_k": 8}\n\n'
        "  grep_context   Full-text search with N surrounding lines.\n"
        '                 {"tool": "grep_context", "pattern": "...", "path": ".", "context_lines": 60}\n\n'
        "  head           Read the first N lines of a file.\n"
        '                 {"tool": "head", "path": "relative/path/File.java", "lines": 200}\n\n'
        "  find           Find files by name glob.\n"
        '                 {"tool": "find", "name_pattern": "*Driver*.java", "path": "."}\n\n'
        "  grep           Discover files containing a pattern (returns file:line matches).\n"
        '                 {"tool": "grep", "pattern": "ClusterDriver", "include": "*.java"}\n\n'
        "## Quality Rules\n"
        "- Every claim must cite specific class names, method signatures, or file:line locations.\n"
        "- If evidence is insufficient for a claim, write \'Evidence insufficient for [claim]\'.\n"
        "- Do NOT fabricate class names, method signatures, API contracts, or runtime behaviour.\n"
        "- The final document must be 500+ words with concrete technical detail.\n"
    )

    def __init__(
        self,
        llm_client: "LLMClient",
        retriever: "HybridRetriever",
        cli_executor: "CliExecutor | None",
        repo_root: Path,
        max_turns: int = 8,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        self.cli_executor = cli_executor
        self.repo_root = repo_root.resolve()
        self.max_turns = max(3, max_turns)
        self.codebase_context: "_CodebaseContext | None" = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def investigate(
        self,
        document: str,
        query: str,
        retrieval_confidence: float,
        rag_hits: "list[RetrievalHit] | None" = None,
    ) -> "tuple[InvestigationTrace, list[str], str]":
        """Run the multi-turn tool loop for *document*.

        Returns a 3-tuple:
        - ``trace``    — :class:`InvestigationTrace` with full command/finding history.
        - ``insights`` — Accumulated memo texts (usable as fallback evidence).
        - ``analysis`` — Complete markdown analysis from ``<final>`` block, or
          ``""`` when the loop exhausted max_turns without a valid ``<final>``.
        """
        memo_history: list[str] = []  # persistent findings — prepended each turn
        # Each tool-result entry: (tool_name, args, output, succeeded, truncated)
        tool_results: list[tuple[str, dict[str, Any], str, bool, bool]] = []
        insights: list[str] = []
        commands_run: list[str] = []

        prompt = self._build_initial_prompt(document, query, rag_hits)
        system = _INVESTIGATOR_SYSTEM + self._TOOL_PROTOCOL

        for turn_idx in range(self.max_turns):
            turns_remaining = self.max_turns - turn_idx - 1
            if turn_idx > 0:
                prompt = self._build_continuation_prompt(
                    memo_history, tool_results, turns_remaining
                )

            try:
                raw = self.llm_client.generate(
                    system, prompt, max_tokens=_SYNTHESIS_MAX_TOKENS
                )
            except Exception as exc:
                _LOG.warning(
                    "doc=%s turn=%d LLM call failed: %s", document, turn_idx, exc
                )
                break

            think = _extract_xml_block(raw, "think")
            memo = _extract_xml_block(raw, "memo")
            tools_text = _extract_xml_block(raw, "tools")
            final_text = _extract_xml_block(raw, "final")

            if think:
                _LOG.debug(
                    "doc=%s turn=%d scratchpad=%d chars (not forwarded)",
                    document, turn_idx, len(think),
                )

            # Accumulate memo; evict oldest entries when the cap is exceeded.
            if memo:
                memo_history.append(memo)
                insights.append(memo)
                total_chars = sum(len(m) for m in memo_history)
                while total_chars > self.MAX_MEMO_TOTAL_CHARS and len(memo_history) > 1:
                    total_chars -= len(memo_history.pop(0))

            # A non-empty <final> block ends the loop immediately.
            if final_text.strip():
                _LOG.info(
                    "doc=%s: multi-turn investigation done in %d turn(s)",
                    document, turn_idx + 1,
                )
                return (
                    self._build_trace(
                        document, query, retrieval_confidence, commands_run, insights
                    ),
                    insights,
                    final_text.strip(),
                )

            # Tool calls — results visible only in the next prompt.
            tool_results = []
            if tools_text:
                for tool_name, args in self._parse_tool_calls(tools_text):
                    output, ok, trunc = self._execute_tool(tool_name, args)
                    commands_run.append(f"{tool_name}({json.dumps(args, separators=(',', ':'))})")
                    if ok and output.strip():
                        insights.append(f"[{tool_name}] {output[:200]}")
                    tool_results.append((tool_name, args, output, ok, trunc))
                    _LOG.debug(
                        "doc=%s turn=%d tool=%s ok=%s chars=%d",
                        document, turn_idx, tool_name, ok, len(output),
                    )

        # Exhausted max_turns — one unconditional forcing pass.
        final_text = self._force_final(document, memo_history)
        return (
            self._build_trace(
                document, query, retrieval_confidence, commands_run, insights
            ),
            insights,
            final_text,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_initial_prompt(
        self,
        document: str,
        query: str,
        rag_hits: "list[RetrievalHit] | None",
    ) -> str:
        topic = document.removesuffix(".md").replace("_", " ")
        synthesis_question = _TOPIC_SYNTHESIS_QUESTIONS.get(
            document,
            f"What are the key architectural and implementation insights about {topic}?",
        )
        codebase_header = ""
        if self.codebase_context:
            codebase_header = (
                f"## Codebase Overview ({self.codebase_context.total_source_files} source files)\n"
                f"{self.codebase_context.module_overview}\n\n"
            )
        rag_block = ""
        if rag_hits:
            rag_lines: list[str] = []
            for hit in rag_hits[:_RAG_CHUNKS_PER_TOPIC]:
                rag_lines.append(
                    f"  [{hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line}]"
                    f" score={hit.score:.2f}\n"
                    f"  {str(hit.chunk.text)[:_RAG_CHUNK_CHARS]}"
                )
            rag_block = (
                "## Seed RAG Hits (use as starting points for tool calls)\n"
                + "\n\n".join(rag_lines)
                + "\n\n"
            )
        return (
            f"{codebase_header}"
            f"## Investigation Topic: {topic}\n"
            f"Seed query: `{query}`\n\n"
            f"## Analysis Question\n{synthesis_question}\n\n"
            f"{rag_block}"
            "## Your Task\n"
            "Use tools to gather comprehensive evidence. When ready, emit a "
            "<final> block containing a complete 500+ word markdown analysis.\n"
        )

    def _build_continuation_prompt(
        self,
        memo_history: list[str],
        tool_results: list[tuple[str, dict[str, Any], str, bool, bool]],
        turns_remaining: int,
    ) -> str:
        """Build the next user-turn prompt.

        Only memo history and the latest batch of tool output are included.
        Earlier tool outputs are deliberately omitted — the LLM is responsible
        for distilling important findings into ``<memo>`` before they are lost.
        """
        parts: list[str] = []

        if memo_history:
            memos_block = "\n\n".join(
                f"[memo/{i + 1}]\n{memo}" for i, memo in enumerate(memo_history)
            )
            parts.append(f"## Your Accumulated Findings\n{memos_block}")

        if tool_results:
            result_blocks: list[str] = []
            for tool_name, args, output, ok, trunc in tool_results:
                args_brief = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
                status = "OK" if ok else "FAILED"
                trunc_note = " [truncated]" if trunc else ""
                result_blocks.append(
                    f"### {tool_name}({args_brief}) [{status}]{trunc_note}\n{output}"
                )
            parts.append("## Latest Tool Results\n" + "\n\n".join(result_blocks))

        if turns_remaining == 0:
            parts.append(
                "**Final turn.** Emit a <final> block now with the complete markdown analysis. "
                "Mark evidence gaps explicitly rather than fabricating."
            )
        elif turns_remaining == 1:
            parts.append(
                f"You have {turns_remaining} turn remaining. Consider emitting <final> soon."
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------

    def _parse_tool_calls(
        self, tools_text: str
    ) -> list[tuple[str, dict[str, Any]]]:
        calls: list[tuple[str, dict[str, Any]]] = []
        for line in tools_text.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
                name = str(obj.pop("tool", "")).strip()
                if name:
                    calls.append((name, obj))
            except json.JSONDecodeError:
                pass
        return calls[: self.MAX_TOOL_CALLS_PER_TURN]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(
        self, tool: str, args: dict[str, Any]
    ) -> tuple[str, bool, bool]:
        """Execute one tool call.  Returns ``(output, succeeded, truncated)``."""
        try:
            if tool == "search":
                raw = self._tool_search(args)
            elif tool == "grep_context":
                raw = self._tool_grep_context(args)
            elif tool == "head":
                raw = self._tool_head(args)
            elif tool == "find":
                raw = self._tool_find(args)
            elif tool == "grep":
                raw = self._tool_grep(args)
            else:
                return (
                    f"Unknown tool '{tool}'. Available: search, grep_context, head, find, grep.",
                    False,
                    False,
                )
            output, trunc = self._truncate(raw)
            return output, True, trunc
        except Exception as exc:
            return f"Execution error: {exc}", False, False

    def _tool_search(self, args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "Error: 'query' is required."
        top_k = max(4, min(24, int(args.get("top_k", 8))))
        hits = self.retriever.search(query, top_k=top_k)
        if not hits:
            return "No results found."
        lines: list[str] = []
        for hit in hits:
            lines.append(
                f"--- {hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line}"
                f" (score={hit.score:.3f}) ---"
            )
            lines.append(str(hit.chunk.text)[:600])
        return "\n".join(lines)

    def _tool_grep_context(self, args: dict[str, Any]) -> str:
        if self.cli_executor is None:
            return "CLI executor not available."
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return "Error: 'pattern' is required."
        path = str(args.get("path", ".")).strip() or "."
        ctx_lines = max(10, min(120, int(args.get("context_lines", 60))))
        result = self.cli_executor.run(["grep_context", pattern, path, str(ctx_lines)])
        return result.stdout if result.succeeded else f"grep_context failed: {result.stderr}"

    def _tool_head(self, args: dict[str, Any]) -> str:
        path = str(args.get("path", "")).strip().lstrip("./").strip()
        if not path:
            return "Error: 'path' is required."
        lines = max(10, min(_FILE_LINES_PER_READ, int(args.get("lines", _FILE_LINES_PER_READ))))
        # Prefer direct read (path-traversal safe); fall back to CLI head.
        safe = self._safe_repo_file(path)
        if safe and safe.is_file():
            try:
                content = safe.read_text(encoding="utf-8", errors="replace")
                return "\n".join(content.splitlines()[:lines])
            except OSError:
                pass
        if self.cli_executor is not None:
            result = self.cli_executor.run(["head", f"-{lines}", path])
            if result.succeeded:
                return result.stdout
        return f"File not accessible: {path}"

    def _tool_find(self, args: dict[str, Any]) -> str:
        if self.cli_executor is None:
            return "CLI executor not available."
        pattern = str(args.get("name_pattern", args.get("pattern", ""))).strip()
        if not pattern:
            return "Error: 'name_pattern' is required."
        path = str(args.get("path", ".")).strip() or "."
        result = self.cli_executor.run(["find", path, "-name", pattern, "-type", "f"])
        return result.stdout if result.succeeded else f"find failed: {result.stderr}"

    def _tool_grep(self, args: dict[str, Any]) -> str:
        if self.cli_executor is None:
            return "CLI executor not available."
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return "Error: 'pattern' is required."
        include = str(args.get("include", "*.java")).strip() or "*.java"
        path = str(args.get("path", ".")).strip() or "."
        result = self.cli_executor.run(
            ["grep", "-rn", f"--include={include}", pattern, path]
        )
        return result.stdout if result.succeeded else f"grep failed: {result.stderr}"

    # ------------------------------------------------------------------
    # Fallback: force final from accumulated memos
    # ------------------------------------------------------------------

    def _force_final(self, document: str, memo_history: list[str]) -> str:
        """One unconditional LLM call demanding a ``<final>`` block."""
        topic = document.removesuffix(".md").replace("_", " ")
        synthesis_question = _TOPIC_SYNTHESIS_QUESTIONS.get(
            document,
            f"What are the key architectural and implementation insights about {topic}?",
        )
        memos_text = (
            "\n\n".join(f"[memo/{i + 1}]\n{m}" for i, m in enumerate(memo_history))
            if memo_history
            else "(no memos written — synthesise from the investigation question and seed evidence)"
        )
        prompt = (
            f"## Topic: {topic}\n"
            f"## Analysis Question\n{synthesis_question}\n\n"
            f"## Accumulated Findings\n{memos_text}\n\n"
            "You MUST emit a <final> block now with the complete markdown analysis. "
            "Structure it as: Executive Summary / Detailed Findings / Design Rationale / "
            "Key Architectural Observations. Mark evidence gaps explicitly."
        )
        try:
            raw = self.llm_client.generate(
                _INVESTIGATOR_SYSTEM + self._TOOL_PROTOCOL,
                prompt,
                max_tokens=_ANALYSIS_MAX_TOKENS,
            )
            final_text = _extract_xml_block(raw, "final")
            # Some models emit bare markdown without the tag; accept it as-is.
            return final_text.strip() if final_text else raw.strip()
        except Exception as exc:
            _LOG.warning("_force_final failed for doc=%s: %s", document, exc)
            return ""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        document: str,
        query: str,
        confidence: float,
        commands_run: list[str],
        insights: list[str],
    ) -> InvestigationTrace:
        topic = document.removesuffix(".md").replace("_", " ")
        memo_count = len([i for i in insights if not i.startswith("[")])
        return InvestigationTrace(
            objective=f"Multi-turn tool-use investigation of '{topic}' seeded by: {query!r}",
            hypothesis=(
                f"Multi-turn loop gathered {len(insights)} finding(s) across "
                f"{len(commands_run)} tool call(s)."
            ),
            known_evidence=tuple(insights[:20]),
            uncertainties=(
                "Static analysis alone cannot confirm runtime behaviour.",
                "Indexed artifacts may not represent the full deployed codebase.",
            ),
            commands_run=tuple(commands_run),
            command_summaries=tuple(commands_run),
            updated_understanding=(
                f"Multi-turn investigation completed {len(commands_run)} tool call(s), "
                f"producing {memo_count} compressed memo block(s). "
                "Findings are combined with vector-retrieved context for final document synthesis."
            ),
            next_investigation_step=(
                "Validate findings against retrieved RAG context and cross-check with "
                "model-driven analysis before high-confidence training use."
            ),
            confidence=confidence,
            iterations=(),
        )

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.MAX_TOOL_OUTPUT_CHARS:
            return text, False
        return text[: self.MAX_TOOL_OUTPUT_CHARS] + "\n\u2026[output truncated]", True

    def _safe_repo_file(self, rel_path: str) -> Path | None:
        """Return an absolute path only if it resolves safely inside *repo_root*."""
        candidate = (self.repo_root / rel_path).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError:
            return None
        return candidate


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
        # Prefer the multi-turn loop when an LLM is available; it drives its
        # own evidence gathering via the <think>/<tools>/<memo>/<final> protocol.
        self._multi_turn_loop: _InvestigatorMultiTurnLoop | None = (
            _InvestigatorMultiTurnLoop(
                llm_client=llm_client,
                retriever=retriever,
                cli_executor=cli_executor,
                repo_root=paths.repository,
            )
            if llm_client is not None and cli_executor is not None else None
        )

    def run(self, resume: bool = True) -> dict[str, int]:
        self.paths.investigator_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: Global Reconnaissance ───────────────────────────────────────
        # Build a shared codebase map from directory structure and build files.
        # Injecting this into every topic investigation avoids each of the 22 topics
        # independently re-discovering the same structural information.
        # Skip reconnaissance only when ALL documents already exist (full resume).
        codebase_context: _CodebaseContext | None = None
        all_done = resume and all(
            (self.paths.investigator_dir / doc).exists() for doc in MANDATORY_DOCUMENTS
        )
        if (self._multi_turn_loop is not None or self._agentic_loop is not None) and not all_done:
            codebase_context = self._build_codebase_context()
            if codebase_context is not None:
                if self._multi_turn_loop is not None:
                    self._multi_turn_loop.codebase_context = codebase_context
                if self._agentic_loop is not None:
                    self._agentic_loop.codebase_context = codebase_context

        # ── Phase 2: Per-topic Investigation ──────────────────────────────────
        written = 0
        for document in MANDATORY_DOCUMENTS:
            target = self.paths.investigator_dir / document
            if resume and target.exists():
                _LOG.info("Skipping already-written document: %s", document)
                written += 1
                continue
            query = _DOCUMENT_QUERIES[document]
            hits = self.retriever.search(query, top_k=_RAG_CHUNKS_PER_TOPIC)
            finding = self._finding_for(document, query, hits)
            trace: InvestigationTrace | None = None
            insights: list[str] = []
            model_analysis: str
            if self._multi_turn_loop is not None:
                # Multi-turn loop drives its own tool use and emits a complete
                # analysis in <final>; skip the separate _model_analysis() call
                # when a non-empty analysis was produced.
                trace, insights, analysis = self._multi_turn_loop.investigate(
                    document, query, finding.confidence, rag_hits=hits
                )
                model_analysis = analysis if analysis else self._model_analysis(
                    document, hits, insights, codebase_context
                )
            elif self._agentic_loop is not None:
                trace, insights = self._agentic_loop.investigate(
                    document, query, finding.confidence, rag_hits=hits
                )
                model_analysis = self._model_analysis(document, hits, insights, codebase_context)
            else:
                model_analysis = self._model_analysis(document, hits, [], codebase_context)
            target.write_text(
                _render_document(document, query, finding, model_analysis, trace),
                encoding="utf-8",
            )
            written += 1
        return {"documents": written}

    def _build_codebase_context(self) -> "_CodebaseContext | None":
        """Run a one-time broad reconnaissance of the repository structure.

        Collects the directory tree and build-file content, then uses the LLM to
        synthesise a compact module overview that is injected into every subsequent
        topic investigation.  Returns ``None`` if no CLI executor is available.
        """
        if self.cli_executor is None:
            return None

        # 1. Directory tree (skip generated/vcs dirs to keep output compact)
        dir_result = self.cli_executor.run([
            "find", ".", "-type", "d",
            "-not", "-path", "*/.git/*",
            "-not", "-path", "*/build/*",
            "-not", "-path", "*/target/*",
            "-not", "-path", "*/.gradle/*",
        ])
        directory_tree = dir_result.stdout[:4_000] if dir_result.succeeded else ""

        # 2. Count source files
        java_count_result = self.cli_executor.run(
            ["find", ".", "-name", "*.java", "-type", "f"]
        )
        total_source_files = (
            len(java_count_result.stdout.splitlines()) if java_count_result.succeeded else 0
        )

        # 3. Read key build files to understand module dependency structure
        build_files_summary = ""
        try:
            build_find = self.cli_executor.run(
                ["find", ".", "-maxdepth", "3", "-name", "build.gradle", "-type", "f"]
            )
            if build_find.succeeded:
                for path in build_find.stdout.splitlines()[:8]:
                    path = path.strip()
                    if not path:
                        continue
                    try:
                        content_res = self.cli_executor.run(["head", "-40", path])
                        if content_res.succeeded and content_res.stdout.strip():
                            build_files_summary += f"\n--- {path} ---\n{content_res.stdout[:600]}\n"
                    except ValueError:
                        pass
        except ValueError:
            pass

        # 4. LLM synthesis: produce a compact module overview from structural evidence
        module_overview = ""
        if self.llm_client and (directory_tree or build_files_summary):
            recon_prompt = (
                f"Repository directory structure:\n{directory_tree}\n\n"
                f"Build files (module definitions):\n{build_files_summary}\n\n"
                f"Total Java source files: {total_source_files}\n\n"
                "Produce a factual 2-3 paragraph overview of this codebase's structure:\n"
                "1. What are the top-level modules/subsystems and their apparent purpose?\n"
                "2. What is the package hierarchy and what domain does each package cover?\n"
                "3. What architectural pattern is used (multi-module monorepo, plugin system, etc.)?\n"
                "Be specific — name the actual module directories you see. Do not speculate.\n"
                "Maximum 300 words."
            )
            try:
                raw = self.llm_client.generate(_RECON_SYSTEM, recon_prompt, max_tokens=_RECON_MAX_TOKENS)
                module_overview = raw.strip() if raw else ""
                if not module_overview:
                    _LOG.warning("LLM recon synthesis returned empty response")
            except Exception as exc:
                _LOG.exception("LLM recon synthesis failed (prompt ~%d chars): %s", len(recon_prompt), exc)

        if not module_overview:
            module_overview = (
                f"Repository contains {total_source_files} Java source files.\n"
                f"Directory layout:\n{directory_tree[:1_500]}"
            )

        return _CodebaseContext(
            directory_tree=directory_tree,
            module_overview=module_overview,
            total_source_files=total_source_files,
            build_files_summary=build_files_summary,
        )

    def _model_analysis(
        self,
        document: str,
        hits: list[RetrievalHit],
        insights: list[str] | None = None,
        codebase_context: "_CodebaseContext | None" = None,
    ) -> str:
        """Return an LLM-generated analysis section, or a placeholder if no endpoint.

        Combines the global codebase context, semantically-retrieved code chunks, and
        agentic investigation insights into a high-quality synthesis.  Token budget is
        calibrated for 128k–256k context models via :data:`_ANALYSIS_MAX_TOKENS`.
        """
        if self.llm_client is None:
            return "Configure an investigator endpoint to enable model-driven analysis."
        synthesis_question = _TOPIC_SYNTHESIS_QUESTIONS.get(
            document,
            f"What are the key architectural and implementation insights about "
            f"{document.removesuffix('.md').replace('_', ' ')} in this codebase?",
        )
        # Global codebase orientation (built once in Phase 1, shared across topics).
        context_section = ""
        if codebase_context:
            context_section = (
                f"CODEBASE OVERVIEW ({codebase_context.total_source_files} source files):\n"
                f"{codebase_context.module_overview}\n\n"
            )
        # RAG-retrieved code chunks.
        evidence_block = "\n".join(
            f"  [{i + 1}] {hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line}\n"
            f"      {hit.chunk.text[:_RAG_CHUNK_CHARS]}"
            for i, hit in enumerate(hits[:_RAG_CHUNKS_PER_TOPIC])
        )
        # Investigation insights from direct source file reading.
        insights_block = ""
        if insights:
            insights_block = (
                f"\n\nINVESTIGATION FINDINGS (from direct source reading across "
                f"{len(insights)} iteration(s)):\n"
                + "\n\n---\n\n".join(f"[Finding {i + 1}]\n{ins}" for i, ins in enumerate(insights))
            )
        user_prompt = (
            f"{context_section}"
            f"ANALYSIS QUESTION:\n{synthesis_question}\n\n"
            f"RETRIEVED CODE CONTEXT (top {len(hits)} semantic matches):\n{evidence_block}"
            f"{insights_block}\n\n"
            "SYNTHESIS INSTRUCTIONS:\n"
            "Produce a comprehensive technical analysis that answers the question above.\n"
            "Structure your response as follows:\n"
            "1. Executive summary (2-3 sentences: what is the core answer?)\n"
            "2. Detailed findings (cite specific classes/methods/algorithms)\n"
            "3. Design rationale (WHY were things designed this way?)\n"
            "4. Key architectural observations (2-3 most important insights)\n"
            "Minimum 500 words. Write for a senior engineer audience."
        )
        try:
            result = self.llm_client.generate(_INVESTIGATOR_SYSTEM, user_prompt, max_tokens=_ANALYSIS_MAX_TOKENS)
        except Exception as exc:
            _LOG.exception(
                "LLM model analysis failed for topic %r (prompt ~%d chars): %s",
                document, len(user_prompt), exc,
            )
            return f"[LLM error] Model analysis failed: {exc}"
        if not result or not result.strip():
            _LOG.warning(
                "LLM model analysis returned empty response for topic %r (prompt ~%d chars)",
                document, len(user_prompt),
            )
            return "[empty response from model — check logs for prompt size or endpoint errors]"
        return result

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
