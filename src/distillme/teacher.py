"""Teacher stage for grounded synthetic instruction generation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

_LOG = logging.getLogger(__name__)

from distillme.investigator import load_findings
from distillme.retrieval import HybridRetriever
from distillme.schemas import DatasetExample, PipelinePaths

if TYPE_CHECKING:
    from distillme.cli_tools import CliExecutor
    from distillme.inference import LLMClient

# Task categories whose examples include a structured investigation trajectory.
_TRAJECTORY_CATEGORIES = frozenset({"agentic_task", "cli_exploration_task", "multi_hop_navigation_task", "retrieval_task"})

TASK_CATEGORIES = (
    "code_understanding",
    "implementation_task",
    "debugging_task",
    "refactoring_task",
    "test_generation",
    "architecture_task",
    "agentic_task",
    "retrieval_task",
    "security_task",
    "performance_task",
    "cli_exploration_task",
    "multi_hop_navigation_task",
)
DIFFICULTIES = ("easy", "medium", "hard", "expert", "multi-hop", "adversarial")
CONFIDENCE_WITH_RETRIEVED_CONTEXT = 0.55
CONFIDENCE_WITHOUT_RETRIEVED_CONTEXT = 0.25

# One question template per (category, difficulty-index) pair — 60 unique prompts.
_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "code_understanding": [
        "Explain the purpose and responsibilities of the code in {files}, citing only the retrieved evidence.",
        "Describe the control flow and branching logic visible in {files}.",
        "What invariants and preconditions does the code in {files} enforce?",
        "How does {files} handle errors and edge cases according to the retrieved source?",
        "Identify the design pattern applied in {files} and provide evidence from the retrieved context.",
        "Trace the data flow through {files} as revealed by the retrieved chunks.",
    ],
    "implementation_task": [
        "Based on conventions in {files}, sketch a new method that follows the same coding style.",
        "Add input validation to {files} consistent with the error-handling patterns in the retrieved code.",
        "Extend {files} to support an additional configuration option following existing conventions.",
        "Implement a factory method for the primary class in {files} matching the retrieved code style.",
        "Add structured logging to {files} following the patterns observed in the retrieved context.",
        "Describe how to implement a caching layer for {files} without violating existing architectural constraints.",
    ],
    "debugging_task": [
        "Identify potential failure modes in {files} based solely on the retrieved code evidence.",
        "What edge cases are NOT handled by the code in {files} according to the retrieved context?",
        "Trace a null-pointer or unchecked-exception risk visible in {files}.",
        "Describe how a concurrency issue could manifest in {files} given the retrieved implementation.",
        "What would cause the code in {files} to throw an unexpected exception?",
        "Identify the root cause of a potential performance degradation in {files}.",
    ],
    "refactoring_task": [
        "Propose a readability improvement to {files} that preserves the existing API contract.",
        "How could the coupling between classes in {files} be reduced based on the retrieved evidence?",
        "Identify a method in {files} that is a candidate for extraction, and explain why.",
        "Describe a performance-neutral refactoring of the conditional logic in {files}.",
        "Propose how to modularise the responsibilities in {files} into smaller focused units.",
        "How would you refactor {files} to improve testability without changing observable behaviour?",
    ],
    "test_generation": [
        "Write a unit-test outline for the primary method in {files} that covers the happy path.",
        "Describe an edge-case test for {files} that exercises the error-handling branch.",
        "What integration-test scenarios would validate the interaction between {files} and its dependencies?",
        "Write a parameterised test outline for {files} covering multiple input variants.",
        "Describe a concurrency test for {files} that detects race conditions visible in the retrieved code.",
        "Outline a property-based test strategy for the data transformations performed by {files}.",
    ],
    "architecture_task": [
        "Describe the architectural role of {files} within the broader service topology.",
        "What service boundaries does {files} define or cross based on the retrieved evidence?",
        "Evaluate the coupling and cohesion tradeoffs visible in {files}.",
        "How does {files} contribute to the layered architecture of the repository?",
        "What scaling constraints does the design of {files} imply?",
        "Propose an alternative architectural pattern for {files} and analyse the tradeoffs.",
    ],
    "agentic_task": [
        "Decompose the task of adding a new feature to {files} into ordered, atomic implementation steps.",
        "Identify all files that would be impacted by a change to the public API of {files}.",
        "Sequence the modifications required to safely refactor {files} without breaking callers.",
        "Plan the implementation of a bug fix in {files}: enumerate affected components and validation steps.",
        "Describe how an agentic system should navigate from {files} to understand the full execution path.",
        "What tool invocations would an autonomous coding agent need to understand and modify {files}?",
    ],
    "retrieval_task": [
        "Locate all files that declare or implement the primary symbol defined in {files}.",
        "Identify every caller of the public methods in {files} within the indexed codebase.",
        "Trace the dependency chain from {files} to its transitive runtime dependencies.",
        "Correlate the test file for {files} and describe what behaviours are verified.",
        "Find all configuration keys consumed by {files} and their default values.",
        "Identify symbol-usage patterns across the repository that mirror {files}.",
    ],
    "security_task": [
        "Identify input-validation weaknesses in {files} based on the retrieved code.",
        "Audit the authentication and authorisation checks visible in {files}.",
        "What injection risks exist in {files} given the retrieved implementation?",
        "Describe how secrets or credentials are handled by {files}.",
        "Evaluate the serialisation and deserialisation risks in {files}.",
        "What security invariant does {files} rely on that is NOT enforced within its own scope?",
    ],
    "performance_task": [
        "Identify the hot path in {files} and describe its computational complexity.",
        "What caching opportunity exists in {files} based on the retrieved code patterns?",
        "Describe how the memory-allocation profile of {files} could be improved.",
        "Identify the database query in {files} most likely to cause a performance bottleneck.",
        "How would you reduce lock contention in {files} based on the retrieved concurrency patterns?",
        "Propose an async-execution refactoring for the blocking call visible in {files}.",
    ],
    "cli_exploration_task": [
        "Use grep to locate all usages of the primary symbol defined in {files} across the repository.",
        "Using find and grep, enumerate every file that transitively imports or depends on {files}.",
        "Construct a git-grep command sequence to trace the change history of the core logic in {files}.",
        "Use grep with a regex pattern to identify all exception types thrown or caught in {files} and their call sites.",
        "Write a find command strategy that discovers all test files associated with {files}.",
        "Describe a grep-based investigation plan to identify every caller of the public API exposed by {files}.",
    ],
    "multi_hop_navigation_task": [
        "Starting from {files}, trace the complete call chain to the persistence or data-access layer.",
        "Follow the data flow from the entry point in {files} through all intermediate transformations to the final output.",
        "Identify the event-driven communication path initiated in {files} and trace it to its ultimate consumer.",
        "Trace the dependency chain from {files} through all transitive dependencies to any external system boundaries.",
        "Follow the exception propagation path originating in {files} through every catch and rethrow boundary.",
        "Reconstruct the multi-component interaction sequence initiated by {files}, identifying each intermediate hop.",
    ],
}

# Per-category chain-of-thought supervision traces.
_REASONING_TRACES: dict[str, str] = {
    "code_understanding": (
        "Step 1: Parse the primary symbols (classes, interfaces, methods) from the retrieved chunks. "
        "Step 2: Identify the entry point and trace the execution path. "
        "Step 3: Note any explicit contracts (throws clauses, Javadoc, annotations). "
        "Step 4: Record unresolved ambiguity where behaviour is inferred rather than stated. "
        "Step 5: Synthesise an explanation bounded strictly by the retrieved evidence."
    ),
    "implementation_task": (
        "Step 1: Extract naming conventions and method signatures from the retrieved code. "
        "Step 2: Identify the error-handling idiom in use (exceptions, Optional, Result). "
        "Step 3: Check for dependency-injection or factory patterns to replicate. "
        "Step 4: Draft the implementation matching the existing style. "
        "Step 5: List validation checks needed before the change is safe to commit."
    ),
    "debugging_task": (
        "Step 1: Identify null-safety, bounds, and precondition violations in the retrieved code. "
        "Step 2: Trace the call path that could trigger the failure. "
        "Step 3: Check test-coverage evidence to assess whether the case is already handled. "
        "Step 4: Propose a minimal, targeted fix that preserves existing contracts. "
        "Step 5: Note uncertainties that require dynamic traces or test execution to confirm."
    ),
    "refactoring_task": (
        "Step 1: Identify coupling, duplication, or responsibility overload in the retrieved code. "
        "Step 2: Verify that the refactoring target is covered by tests (test-coverage evidence). "
        "Step 3: Enumerate the public API surface that must remain stable. "
        "Step 4: Propose the refactoring in small, reviewable steps. "
        "Step 5: List regression risks and required test updates."
    ),
    "test_generation": (
        "Step 1: Enumerate the distinct execution paths in the retrieved code. "
        "Step 2: Identify existing test coverage and gaps. "
        "Step 3: Choose an appropriate test scope (unit, integration, property-based). "
        "Step 4: Design assertions that verify observable outcomes rather than implementation details. "
        "Step 5: Note any infrastructure or mocking requirements visible in the retrieved test evidence."
    ),
    "architecture_task": (
        "Step 1: Place the retrieved code within the package and layer hierarchy. "
        "Step 2: Identify inbound and outbound dependencies from the graph evidence. "
        "Step 3: Evaluate cohesion and coupling against the retrieved conventions. "
        "Step 4: Assess architectural consistency with the dominant patterns in the index. "
        "Step 5: Propose improvements with explicit tradeoff analysis."
    ),
    "agentic_task": (
        "Step 1: Decompose the objective into an ordered, atomic task list. "
        "Step 2: Use retrieval to identify all impacted files and symbols. "
        "Step 3: Sequence modifications to minimise intermediate broken states. "
        "Step 4: Enumerate validation steps (compile, test, lint) for each sub-task. "
        "Step 5: Flag uncertainty items that require human review before continuation."
    ),
    "retrieval_task": (
        "Step 1: Formulate targeted search queries for the symbols in question. "
        "Step 2: Expand the query using call-graph and dependency-graph edges. "
        "Step 3: Rank results by relevance to the specific retrieval objective. "
        "Step 4: Validate that every retrieved reference corresponds to an indexed artifact. "
        "Step 5: Summarise the retrieval chain for downstream agent consumption."
    ),
    "security_task": (
        "Step 1: Enumerate all external input surfaces in the retrieved code. "
        "Step 2: Check for missing validation, sanitisation, and encoding. "
        "Step 3: Audit authentication/authorisation entry points visible in the index. "
        "Step 4: Assess serialisation and secret-handling patterns. "
        "Step 5: Report findings with confidence scores and unresolved uncertainties."
    ),
    "performance_task": (
        "Step 1: Identify the hot execution path from retrieved code and call-graph evidence. "
        "Step 2: Profile algorithmic complexity of loops, recursion, and query patterns. "
        "Step 3: Look for caching and memoisation opportunities. "
        "Step 4: Check threading and async patterns for contention or blocking. "
        "Step 5: Propose optimisations ranked by expected impact, with risk assessment."
    ),
    "cli_exploration_task": (
        "Step 1: Identify the primary symbols defined in the target files using grep or ctags. "
        "Step 2: Construct targeted grep commands to locate all usages across the repository. "
        "Step 3: Expand the search to callers, implementors, and related configuration files. "
        "Step 4: Use git grep to trace the symbol through version history if change tracking is needed. "
        "Step 5: Summarise the discovered symbol topology, noting areas unreachable by the current index."
    ),
    "multi_hop_navigation_task": (
        "Step 1: Identify the entry point and initial call or event emission in the source files. "
        "Step 2: Follow the first hop to locate the immediate dependency or consumer. "
        "Step 3: Continue tracing each hop, recording intermediate files and symbols at every step. "
        "Step 4: Identify boundary crossings — service, layer, or module transitions. "
        "Step 5: Produce a complete hop-by-hop chain with a confidence score for each link."
    ),
}

# Per-category answer framings (deterministic baseline; real LLM replaces this).
_ANSWER_FRAMINGS: dict[str, str] = {
    "code_understanding": (
        "Based on the retrieved evidence, {files} serves the following purpose: "
        "start from the cited source locations, apply the reasoning trace above, and report only "
        "what is directly observable. Mark any inferred or uncertain claims explicitly."
    ),
    "implementation_task": (
        "To extend {files} consistently, follow the naming and error-handling conventions "
        "observable in the retrieved chunks. Validate the change by compiling and running "
        "the linked tests. Uncertain about behaviour not covered by retrieved evidence."
    ),
    "debugging_task": (
        "The failure mode in {files} most supported by the retrieved evidence is indicated "
        "by the precondition and branching patterns in the cited lines. "
        "Uncertain about runtime paths not present in the indexed source."
    ),
    "refactoring_task": (
        "The refactoring of {files} should proceed in small, test-verified steps. "
        "Preserve the public API surface visible in the retrieved chunks. "
        "Uncertain about callers outside the indexed scope."
    ),
    "test_generation": (
        "The test suite for {files} should cover the execution paths visible in the retrieved "
        "chunks: the happy path, the null/empty input branch, and any exception path. "
        "Uncertain about integration boundaries not present in the indexed artifacts."
    ),
    "architecture_task": (
        "The architectural role of {files} is constrained by the dependency and call-graph "
        "evidence in the retrieved context. Scaling or restructuring decisions should preserve "
        "the service boundaries visible in the index. Uncertain about runtime topology."
    ),
    "agentic_task": (
        "The implementation plan for {files} decomposes into the steps in the reasoning trace. "
        "Each step must be validated before proceeding to the next. "
        "Uncertain about impacted files outside the retrieved scope."
    ),
    "retrieval_task": (
        "The retrieval result for {files} is bounded by the indexed artifacts. "
        "Expand the query using the call-graph edges in the index for broader coverage. "
        "Uncertain about symbols absent from the current index."
    ),
    "security_task": (
        "The security posture of {files} as observable in the retrieved code shows the following "
        "concerns: review each input surface in the cited lines. "
        "Uncertain about runtime injection vectors not present in the indexed source."
    ),
    "performance_task": (
        "The performance characteristics of {files} visible in the retrieved chunks suggest "
        "the hot path and caching opportunities noted in the reasoning trace. "
        "Uncertain about runtime profiling data not present in the indexed artifacts."
    ),
    "cli_exploration_task": (
        "The grep-based exploration of {files} reveals the following symbol topology: "
        "start from the primary definitions in the cited files and trace all usage sites "
        "confirmed by the retrieved evidence. Flag any symbol usages outside the indexed scope as uncertain."
    ),
    "multi_hop_navigation_task": (
        "The multi-hop navigation from {files} follows the path: "
        "entry point → [intermediate hops] → final consumer, as reconstructed from the retrieved evidence. "
        "Each hop is bounded by indexed artifacts; cross-service hops outside the index are marked uncertain."
    ),
}

_FALLBACK_ANSWER = (
    "Insufficient retrieved evidence is available for this query. "
    "Defer to additional indexing or retrieval before making repository-specific claims."
)


class TeacherAgent:
    """Transforms investigator outputs into source-grounded dataset records."""

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

    def run(self, resume: bool = True) -> dict[str, int]:
        self.paths.dataset_dir.mkdir(parents=True, exist_ok=True)
        findings = load_findings(self.paths.investigator_dir)

        partial_path = self.paths.dataset_dir / "instruction_dataset.partial.jsonl"
        final_path = self.paths.dataset_dir / "instruction_dataset.jsonl"

        # Resume: recover task_ids already written during a previous interrupted run.
        completed_ids: frozenset[str] = frozenset()
        if resume and partial_path.exists():
            ids: set[str] = set()
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
            if ids:
                completed_ids = frozenset(ids)
                _LOG.warning("Resuming teacher stage: %d examples already written.", len(completed_ids))
        elif not resume and partial_path.exists():
            # Explicit restart: discard any leftover partial work.
            partial_path.unlink()

        total = len(completed_ids)
        with partial_path.open("a", encoding="utf-8") as handle:
            for example in self._examples(findings, skip_ids=completed_ids):
                handle.write(json.dumps(example.to_jsonable(), sort_keys=True) + "\n")
                handle.flush()  # each example survives an interrupt
                total += 1

        # Atomic rename so the final file is always a complete dataset.
        partial_path.replace(final_path)

        manifest = {
            "schema": "distillme.dataset.v1",
            "examples": total,
            "categories": list(TASK_CATEGORIES),
            "difficulties": list(DIFFICULTIES),
            "quality_controls": [
                "source_grounding_verification",
                "symbol_existence_validation",
                "retrieval_consistency_checks",
                "hallucination_detection",
                "contradiction_detection",
                "duplicate_elimination",
                "semantic_diversity_scoring",
                "difficulty_balancing",
                "coverage_analysis",
            ],
        }
        (self.paths.dataset_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return {"examples": total}

    def _examples(
        self,
        findings: list[dict[str, str]],
        skip_ids: frozenset[str] = frozenset(),
    ):
        """Yield one DatasetExample per (category × difficulty) pair.

        ``skip_ids`` contains task_ids already persisted in a partial file;
        those iterations are skipped so only genuinely new examples are yielded.
        ``global_index`` is always incremented even for skipped items to keep
        task_id numbering stable across interrupted and resumed runs.
        """
        finding_cycle = findings if findings else [{"path": "placeholder_no_findings", "text": ""}]
        global_index = 0
        for category in TASK_CATEGORIES:
            templates = _QUESTION_TEMPLATES[category]
            reasoning = _REASONING_TRACES[category]
            for diff_idx, difficulty in enumerate(DIFFICULTIES):
                task_id = f"{category}-{difficulty}-{global_index:05d}"
                finding = finding_cycle[global_index % len(finding_cycle)]
                global_index += 1  # always increment so task_ids stay stable on resume
                if task_id in skip_ids:
                    continue
                query = f"{category} {difficulty} {finding['path'].removesuffix('.md')}"
                hits = self.retriever.search(query, top_k=4)
                contexts = [hit.to_context() for hit in hits]
                if not contexts:
                    contexts = self._fallback_context()
                supporting_files = sorted({str(ctx["path"]) for ctx in contexts})
                symbols = sorted({sym for ctx in contexts for sym in ctx.get("symbols", [])})
                question = _build_question(templates[diff_idx], supporting_files)
                answer = _build_answer(category, supporting_files, self.llm_client)
                confidence = (
                    CONFIDENCE_WITH_RETRIEVED_CONTEXT if contexts else CONFIDENCE_WITHOUT_RETRIEVED_CONTEXT
                )
                investigation_trace = ""
                if category in _TRAJECTORY_CATEGORIES:
                    investigation_trace = _build_investigation_trace(
                        category, supporting_files, self.cli_executor
                    )
                yield DatasetExample(
                    task_id=task_id,
                    task_category=category,
                    difficulty=difficulty,
                    repository_context=f"Investigator document: {finding['path']}",
                    retrieved_context=contexts,
                    question=question,
                    reasoning_trace=reasoning,
                    answer=answer,
                    supporting_files=supporting_files,
                    symbols=symbols,
                    architectural_constraints=[
                        "Cite retrieved evidence for every repository-specific claim.",
                        "Preserve uncertainty when source evidence is incomplete.",
                        "Validate symbols against the index before recommending code changes.",
                    ],
                    validation_checks=[
                        "supporting_files_exist",
                        "retrieved_context_non_empty_when_available",
                        "answer_contains_uncertainty_guardrail",
                    ],
                    negative_examples=[
                        "Do not invent classes, APIs, runtime behaviour, or architectural intent "
                        "absent from the retrieved context."
                    ],
                    confidence=confidence,
                    investigation_trace=investigation_trace,
                )

    def _fallback_context(self) -> list[dict[str, object]]:
        if not self.retriever.chunks:
            return []
        chunk = self.retriever.chunks[0]
        return [
            {
                "path": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbols": list(chunk.symbols),
                "score": 0.0,
                "text": chunk.text,
                "reasons": ["fallback_grounding_context"],
            }
        ]


def _build_question(template: str, supporting_files: list[str]) -> str:
    files = ", ".join(supporting_files[:3]) if supporting_files else "the retrieved repository context"
    return template.format(files=files)


def _build_answer(
    category: str,
    supporting_files: list[str],
    llm_client: "LLMClient | None",
) -> str:
    if not supporting_files:
        return _FALLBACK_ANSWER
    framing = _ANSWER_FRAMINGS[category]
    files = ", ".join(supporting_files[:3])
    deterministic = framing.format(files=files)
    # When a real (non-stub) HTTP endpoint is configured, delegate answer generation.
    if llm_client is not None:
        from distillme.inference import HttpLLMClient  # avoid circular import at module level

        if isinstance(llm_client, HttpLLMClient):
            system = (
                "You are a repository-specialist coding advisor. "
                "Answer only from the retrieved source evidence provided. "
                "Mark any uncertain or inferred claims explicitly."
            )
            user = f"Category: {category}\nFiles: {files}\nTask: {deterministic}"
            try:
                return llm_client.generate(system, user)
            except RuntimeError:
                pass
    return deterministic


def _build_investigation_trace(
    category: str,
    supporting_files: list[str],
    cli_executor: "CliExecutor | None",
) -> str:
    """Build a structured investigation trajectory string for agentic dataset examples.

    When *cli_executor* is available the trace includes real command output;
    otherwise it falls back to a deterministic template that still demonstrates
    the expected trajectory format.
    """
    files_label = ", ".join(supporting_files[:2]) if supporting_files else "the target files"
    cli_section_lines: list[str] = []

    # Run representative CLI commands when an executor is available.
    if cli_executor is not None:
        probe_commands: list[list[str]] = []
        if category in {"cli_exploration_task", "agentic_task"}:
            probe_commands = [
                ["find", ".", "-name", "*.java", "-type", "f"],
                ["grep", "-r", "--include=*.java", "-l", "class", "."],
            ]
        elif category in {"multi_hop_navigation_task", "retrieval_task"}:
            probe_commands = [
                ["find", ".", "-type", "f", "-name", "*.java"],
                ["grep", "-r", "--include=*.java", "-n", "import", "."],
            ]
        for args in probe_commands:
            try:
                result = cli_executor.run(args)
                cli_section_lines.append(f"COMMAND: `{result.command}`")
                cli_section_lines.append(f"OUTPUT SUMMARY:\n{result.summary()}")
            except ValueError:
                pass

    cli_block = "\n".join(cli_section_lines) if cli_section_lines else "CLI executor not configured; trajectory is template-based."
    return (
        f"OBJECTIVE: Investigate {category.replace('_', ' ')} patterns in {files_label}.\n\n"
        f"CURRENT HYPOTHESIS: The target files contain {category.replace('_', ' ')} "
        "patterns that can be confirmed through iterative CLI exploration and RAG retrieval.\n\n"
        "KNOWN EVIDENCE:\n"
        f"  - Retrieved context chunks reference: {files_label}\n"
        "  - Symbol definitions extracted from indexed artifacts\n\n"
        "UNCERTAINTIES:\n"
        "  - Runtime behaviour not observable from static source alone\n"
        "  - Cross-service interactions may extend beyond indexed artifacts\n\n"
        f"{cli_block}\n\n"
        "UPDATED UNDERSTANDING: Combined CLI exploration and vector retrieval provide "
        "grounded evidence for this investigation; remaining gaps require dynamic analysis.\n\n"
        "NEXT INVESTIGATION STEP: Cross-validate findings against the full call graph and "
        "inspect test coverage for the identified symbols."
    )
