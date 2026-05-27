"""Teacher stage that compiles investigator-guided curriculum artifacts."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

_LOG = logging.getLogger(__name__)

from distillme.investigator import load_findings
from distillme.retrieval import RetrievalHit
from distillme.schemas import (
    CurriculumSection,
    DatasetExample,
    InvestigatorFindingRecord,
    PipelinePaths,
    TopicUnit,
)

if TYPE_CHECKING:
    from distillme.cli_tools import CliExecutor
    from distillme.inference import LLMClient
    from distillme.retrieval import ChromaRetriever, HybridRetriever

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
DIFFICULTIES = ("easy", "medium", "hard", "expert")
HIGH_CONFIDENCE_THRESHOLD = 0.78

_SOURCE_REF_RE = re.compile(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?(?:\s+score=.*)?$")
_GREP_CONTEXT_HEADER_RE = re.compile(r"^===\s+(.+?):(\d+)\s+===$")
_SYMBOL_DEF_RE = re.compile(r"\b(class|def|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
_PATH_IN_TEXT_RE = re.compile(
    r"([A-Za-z0-9_./-]+\.(?:py|java|kt|go|ts|js|scala|cs|rb|toml|json|ya?ml|md))"
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_xml_block(text: str, tag: str) -> str:
    """Return the content of the first ``<tag>…</tag>`` block in *text*, stripped.

    Returns an empty string when the tag is absent.  Compiled inline to avoid
    a global cache dict; these four tags are called once per turn per loop.
    """
    pattern = re.compile(rf"<{re.escape(tag)}>([\s\S]*?)</{re.escape(tag)}>", re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


_TEMPLATE_TRACK = {
    "concept_card": "concept",
    "api_contract_card": "concept",
    "execution_flow_trace": "concept",
    "retrieval_and_tool_use_task": "action",
    "implementation_plan_task": "action",
    "debugging_and_failure_analysis_task": "action",
    "test_design_task": "action",
    "contrastive_negative_example": "uncertainty",
    "ambiguity_and_deferral_example": "uncertainty",
    "design_deliberation_trace": "reasoning",
    "best_practice_transfer_card": "reasoning",
    "api_selection_debate": "reasoning",
}

_TEMPLATE_TO_CATEGORY = {
    "concept_card": "code_understanding",
    "api_contract_card": "architecture_task",
    "execution_flow_trace": "multi_hop_navigation_task",
    "retrieval_and_tool_use_task": "cli_exploration_task",
    "implementation_plan_task": "implementation_task",
    "debugging_and_failure_analysis_task": "debugging_task",
    "test_design_task": "test_generation",
    "contrastive_negative_example": "retrieval_task",
    "ambiguity_and_deferral_example": "retrieval_task",
    "design_deliberation_trace": "agentic_task",
    "best_practice_transfer_card": "refactoring_task",
    "api_selection_debate": "agentic_task",
}


@dataclass(frozen=True)
class _SectionSpec:
    section_id: str
    section_objective: str
    prerequisite_sections: tuple[str, ...]
    agenda_topics: tuple[str, ...]
    seed_terms: tuple[str, ...]
    recommended_task_families: tuple[str, ...]


_SECTION_SPECS: tuple[_SectionSpec, ...] = (
    _SectionSpec(
        section_id="repository_topology_and_module_boundaries",
        section_objective="Map repository topology, module boundaries, and ownership seams.",
        prerequisite_sections=(),
        agenda_topics=("architecture_overview", "package_relationships", "service_catalog", "dependency_analysis"),
        seed_terms=("module", "package", "boundary", "layer", "dependency"),
        recommended_task_families=("concept_card", "retrieval_and_tool_use_task", "best_practice_transfer_card"),
    ),
    _SectionSpec(
        section_id="shared_contracts_and_api_surfaces",
        section_objective="Extract shared contracts, API surfaces, and lifecycle expectations.",
        prerequisite_sections=("repository_topology_and_module_boundaries",),
        agenda_topics=("api_behavior", "service_catalog", "domain_model", "coding_conventions"),
        seed_terms=("interface", "contract", "hook", "method", "signature"),
        recommended_task_families=("api_contract_card", "api_selection_debate", "design_deliberation_trace"),
    ),
    _SectionSpec(
        section_id="execution_and_event_flows",
        section_objective="Reconstruct request, event, and dispatcher execution flows.",
        prerequisite_sections=(
            "repository_topology_and_module_boundaries",
            "shared_contracts_and_api_surfaces",
        ),
        agenda_topics=("execution_flows", "event_flow_analysis", "state_machine_analysis"),
        seed_terms=("flow", "handler", "dispatch", "event", "transition"),
        recommended_task_families=("execution_flow_trace", "retrieval_and_tool_use_task", "implementation_plan_task"),
    ),
    _SectionSpec(
        section_id="configuration_and_state",
        section_objective="Explain configuration keys, defaults, and state handling.",
        prerequisite_sections=("repository_topology_and_module_boundaries",),
        agenda_topics=("configuration_model", "database_model", "operational_behavior", "caching_behavior"),
        seed_terms=("config", "state", "default", "manifest", "cache"),
        recommended_task_families=("concept_card", "api_contract_card", "implementation_plan_task"),
    ),
    _SectionSpec(
        section_id="error_and_exception_behavior",
        section_objective="Document exception taxonomy, failure behavior, and guardrails.",
        prerequisite_sections=("shared_contracts_and_api_surfaces", "execution_and_event_flows"),
        agenda_topics=("exception_taxonomy", "anti_patterns", "execution_flows"),
        seed_terms=("exception", "error", "raise", "throw", "retry"),
        recommended_task_families=("debugging_and_failure_analysis_task", "contrastive_negative_example", "test_design_task"),
    ),
    _SectionSpec(
        section_id="security_and_operational_behavior",
        section_objective="Capture security controls and operational behavior constraints.",
        prerequisite_sections=("shared_contracts_and_api_surfaces", "configuration_and_state"),
        agenda_topics=("security_analysis", "operational_behavior", "configuration_model"),
        seed_terms=("security", "auth", "validation", "operations", "logging"),
        recommended_task_families=("concept_card", "debugging_and_failure_analysis_task", "best_practice_transfer_card"),
    ),
    _SectionSpec(
        section_id="concurrency_and_performance",
        section_objective="Extract concurrency patterns, hot paths, and performance constraints.",
        prerequisite_sections=("execution_and_event_flows",),
        agenda_topics=("concurrency_patterns", "performance_characteristics", "caching_behavior"),
        seed_terms=("thread", "async", "lock", "latency", "cache"),
        recommended_task_families=("execution_flow_trace", "implementation_plan_task", "debugging_and_failure_analysis_task"),
    ),
    _SectionSpec(
        section_id="testing_strategy_and_validation_patterns",
        section_objective="Map testing strategy, validation patterns, and regression coverage.",
        prerequisite_sections=("shared_contracts_and_api_surfaces", "configuration_and_state"),
        agenda_topics=("testing_strategy", "exception_taxonomy", "coding_conventions"),
        seed_terms=("test", "assert", "fixture", "validation", "regression"),
        recommended_task_families=("test_design_task", "debugging_and_failure_analysis_task", "best_practice_transfer_card"),
    ),
    _SectionSpec(
        section_id="agentic_navigation_and_retrieval_workflows",
        section_objective="Train retrieval, CLI navigation, and source-grounded exploration workflows.",
        prerequisite_sections=(
            "repository_topology_and_module_boundaries",
            "execution_and_event_flows",
        ),
        agenda_topics=("architecture_overview", "execution_flows", "dependency_analysis"),
        seed_terms=("grep", "find", "retrieval", "symbol", "callers"),
        recommended_task_families=("retrieval_and_tool_use_task", "api_selection_debate", "design_deliberation_trace"),
    ),
    _SectionSpec(
        section_id="implementation_and_refactoring_planning",
        section_objective="Compile implementation and refactoring planning patterns.",
        prerequisite_sections=(
            "shared_contracts_and_api_surfaces",
            "testing_strategy_and_validation_patterns",
            "agentic_navigation_and_retrieval_workflows",
        ),
        agenda_topics=("coding_conventions", "anti_patterns", "testing_strategy", "service_catalog"),
        seed_terms=("refactor", "plan", "extend", "impact", "validation"),
        recommended_task_families=("implementation_plan_task", "design_deliberation_trace", "api_selection_debate"),
    ),
    _SectionSpec(
        section_id="adversarial_and_uncertainty_heavy_cases",
        section_objective="Teach hallucination resistance, contrastive reasoning, and safe deferral.",
        prerequisite_sections=(
            "security_and_operational_behavior",
            "agentic_navigation_and_retrieval_workflows",
            "implementation_and_refactoring_planning",
        ),
        agenda_topics=("anti_patterns", "exception_taxonomy", "security_analysis", "glossary"),
        seed_terms=("uncertainty", "contradiction", "unsupported", "defer", "ambiguity"),
        recommended_task_families=("contrastive_negative_example", "ambiguity_and_deferral_example", "design_deliberation_trace"),
    ),
)


@dataclass
class _SectionAccumulator:
    core_files: set[str]
    core_symbols: set[str]
    api_contracts: set[str]
    recurring_conventions: set[str]
    best_practices: set[str]
    invariants: set[str]
    execution_flows: set[str]
    failure_modes: set[str]
    evidence_refs: set[str]
    ambiguities: set[str]
    commands_executed: set[str]
    retrieved_contexts: list[dict[str, object]]
    investigator_topics: set[str]
    iterations: int


# ---------------------------------------------------------------------------
# Multi-turn tool-use orchestrator for LLM-driven Q&A synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ToolCall:
    """A single tool invocation requested by the LLM in one response turn."""

    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class _ToolResult:
    """Result of executing one tool call."""

    tool: str
    args: dict[str, Any]
    output: str
    succeeded: bool
    truncated: bool = False


@dataclass
class _TurnParsed:
    """Parsed content of one LLM response.

    ``think``        Private scratchpad.  Logged but **never** injected into future prompts.
    ``memo``         Persistent findings.  Carried forward in every continuation prompt.
    ``tool_calls``   Requests to execute; results visible only in the *next* turn.
    ``final_output`` Terminal JSON block; signals loop completion.
    ``is_final``     True when a non-empty ``<final>`` block was found.
    """

    think: str
    memo: str
    tool_calls: list[_ToolCall]
    final_output: str
    is_final: bool


class _MultiTurnSynthesizer:
    """Drives a multi-turn, tool-equipped loop for evidence-grounded Q&A synthesis.

    Context-management design
    -------------------------
    Each conversation turn the LLM emits up to four XML-delimited sections:

    ``<think>``
        Private scratchpad reasoning — stripped from all subsequent prompts.
        The LLM writes freely here without polluting the accumulated context
        window.  Mirrors Claude's extended-thinking blocks and Aider's
        edit-scratchpad: intermediate reasoning stays local to each turn.

    ``<tools>``
        One JSON tool-call per line.  The orchestrator executes them and
        injects results **only** into the immediately following prompt.  After
        that single turn the raw output is discarded — the LLM is expected to
        distil key findings into ``<memo>`` instead.

    ``<memo>``
        Compressed, confirmed findings the LLM wants to preserve.  Every memo
        is appended to a rolling history and prepended to each continuation
        prompt.  Total memo history is capped at ``MAX_MEMO_TOTAL_CHARS`` to
        bound context growth — oldest entries are dropped when the cap is hit,
        following the same eviction strategy used by SWE-agent's observation
        compressor.

    ``<final>``
        Terminal JSON block with all 12 template-family Q&A pairs.  Presence
        of a non-empty block ends the loop immediately.

    Available tools
    ---------------
    ``search``         Semantic search over the indexed code chunks.
    ``grep_context``   Full-text grep with surrounding lines via :class:`CliExecutor`.
    ``head``           Read the first N lines of a source file.
    ``find``           Find files matching a name glob.
    """

    MAX_TOOL_CALLS_PER_TURN: int = 4
    MAX_TOOL_OUTPUT_CHARS: int = 3500
    MAX_MEMO_TOTAL_CHARS: int = 6000

    _SYSTEM_PROMPT: str = (
        "You are a curriculum compiler with tool access to a Java source repository.\n"
        "Generate concrete, evidence-grounded training examples for a 7B coding model.\n"
        "\n"
        "## Available Tools\n"
        "Invoke tools by writing one JSON object per line inside <tools> tags:\n"
        "\n"
        "  search         Semantic search over indexed code chunks.\n"
        '                 {"tool": "search", "query": "...", "top_k": 8}\n'
        "\n"
        "  grep_context   Full-text search with surrounding context.\n"
        '                 {"tool": "grep_context", "pattern": "...", "path": ".", "context_lines": 60}\n'
        "\n"
        "  head           Read the first N lines of a source file.\n"
        '                 {"tool": "head", "path": "relative/path/File.java", "lines": 200}\n'
        "\n"
        "  find           Find files matching a name glob pattern.\n"
        '                 {"tool": "find", "name_pattern": "*Driver*.java", "path": "."}\n'
        "\n"
        "## Response Format — use EXACTLY these XML tags every turn\n"
        "\n"
        "<think>\n"
        "Private reasoning — what to search for next, hypotheses, coverage gaps.\n"
        "STRIPPED from all subsequent prompts. Write freely.\n"
        "</think>\n"
        "\n"
        "<tools>\n"
        '{"tool": "search", "query": "connection pool init", "top_k": 8}\n'
        '{"tool": "grep_context", "pattern": "ClusterDriver", "path": ".", "context_lines": 40}\n'
        "</tools>\n"
        "\n"
        "<memo>\n"
        "Confirmed findings — cite specific class/method/file names from tool output.\n"
        "PERSISTS across all subsequent turns. Be concise. Omit if nothing new.\n"
        "</memo>\n"
        "\n"
        "When you have enough evidence for all 12 template families, emit ONLY this:\n"
        "\n"
        "<final>\n"
        "{\n"
        '  "concept_card":                        {"question": "...", "answer": "..."},\n'
        '  "api_contract_card":                   {"question": "...", "answer": "..."},\n'
        '  "execution_flow_trace":                {"question": "...", "answer": "..."},\n'
        '  "retrieval_and_tool_use_task":         {"question": "...", "answer": "..."},\n'
        '  "implementation_plan_task":            {"question": "...", "answer": "..."},\n'
        '  "debugging_and_failure_analysis_task": {"question": "...", "answer": "..."},\n'
        '  "test_design_task":                    {"question": "...", "answer": "..."},\n'
        '  "contrastive_negative_example":        {"question": "...", "answer": "..."},\n'
        '  "ambiguity_and_deferral_example":      {"question": "...", "answer": "..."},\n'
        '  "design_deliberation_trace":           {"question": "...", "answer": "..."},\n'
        '  "best_practice_transfer_card":         {"question": "...", "answer": "..."},\n'
        '  "api_selection_debate":                {"question": "...", "answer": "..."}\n'
        "}\n"
        "</final>\n"
        "\n"
        "## Quality Rules\n"
        "- Every claim must cite specific class names, method names, or file paths from tool results.\n"
        "- If evidence is insufficient for a claim, write 'Evidence insufficient for [claim]'.\n"
        "- The question must be a genuine learning question a developer would ask.\n"
        "- The answer must be complete, self-contained, and evidence-grounded.\n"
        "- Do NOT fabricate APIs or behaviors not observed in tool output.\n"
    )

    def __init__(
        self,
        llm_client: "LLMClient",
        retriever: "HybridRetriever | ChromaRetriever",
        cli_executor: "CliExecutor | None",
        repo_root: Path,
        max_turns: int = 6,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        self.cli_executor = cli_executor
        self.repo_root = repo_root.resolve()
        self.max_turns = max(2, max_turns)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def synthesize(self, section: CurriculumSection) -> dict[str, tuple[str, str]]:
        """Run the tool-use loop; return ``{template_family: (question, answer)}``.

        The loop terminates as soon as the model emits a ``<final>`` block or
        after ``max_turns`` turns, whichever comes first.  On exhaustion a
        final forcing pass is attempted using the accumulated memo history.
        """
        memo_history: list[str] = []          # persistent findings — prepended each turn
        tool_results: list[_ToolResult] = []  # latest batch only — dropped after one turn

        prompt = self._build_initial_prompt(section)

        for turn_idx in range(self.max_turns):
            turns_remaining = self.max_turns - turn_idx - 1

            if turn_idx > 0:
                prompt = self._build_continuation_prompt(memo_history, tool_results, turns_remaining)

            try:
                raw = self.llm_client.generate(self._SYSTEM_PROMPT, prompt, max_tokens=3000)
            except Exception as exc:
                _LOG.warning(
                    "section=%s turn=%d LLM call failed: %s", section.section_id, turn_idx, exc
                )
                break

            parsed = self._parse_turn(raw)

            # <think> is private — log character count only so we can diagnose
            # without reinjecting the content into later turns.
            if parsed.think:
                _LOG.debug(
                    "section=%s turn=%d scratchpad=%d chars (not forwarded)",
                    section.section_id,
                    turn_idx,
                    len(parsed.think),
                )

            # Accumulate memo; evict oldest entries when the cap is exceeded.
            if parsed.memo:
                memo_history.append(parsed.memo)
                total_chars = sum(len(m) for m in memo_history)
                while total_chars > self.MAX_MEMO_TOTAL_CHARS and len(memo_history) > 1:
                    total_chars -= len(memo_history.pop(0))

            # A valid <final> block ends the loop immediately.
            if parsed.is_final:
                result = self._parse_final_output(parsed.final_output)
                if result:
                    _LOG.info(
                        "section=%s: synthesis done in %d turn(s), families=%d",
                        section.section_id,
                        turn_idx + 1,
                        len(result),
                    )
                    return result
                _LOG.warning(
                    "section=%s: malformed <final> at turn %d, continuing",
                    section.section_id,
                    turn_idx,
                )

            # Tool results are only injected into the next prompt, then dropped.
            tool_results = (
                self._execute_tool_calls(parsed.tool_calls, section.section_id, turn_idx)
                if parsed.tool_calls
                else []
            )

        # Exhausted max_turns — one forcing pass from accumulated memos.
        return self._force_final(section, memo_history)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_initial_prompt(self, section: CurriculumSection) -> str:
        symbols_line = ", ".join(list(section.core_symbols)[:12])
        files_block = "\n".join(list(section.core_files)[:8])
        contracts_block = "\n".join(list(section.api_contracts)[:10])
        flows_block = "\n".join(list(section.execution_flows)[:8])
        return (
            f"## Section: {section.section_id}\n"
            f"Objective: {section.section_objective}\n\n"
            "## Seed Evidence (from prior investigation)\n"
            f"Key symbols: {symbols_line}\n\n"
            f"Key files:\n{files_block}\n\n"
            f"Observed API contracts:\n{contracts_block}\n\n"
            f"Execution flow observations:\n{flows_block}\n\n"
            "## Your Task\n"
            "Use tools to deepen your understanding, then emit all 12 template families "
            "in a single <final> block when ready.\n"
        )

    def _build_continuation_prompt(
        self,
        memo_history: list[str],
        tool_results: list[_ToolResult],
        turns_remaining: int,
    ) -> str:
        """Build the next user-turn prompt from persistent memos and latest tool output.

        Only the memo history and the immediately preceding tool results are
        included — earlier tool outputs are deliberately omitted to prevent
        context bloat.  The LLM is responsible for distilling important
        findings into ``<memo>`` before they are lost.
        """
        parts: list[str] = []

        if memo_history:
            memos_block = "\n\n".join(
                f"[memo/{i + 1}]\n{memo}" for i, memo in enumerate(memo_history)
            )
            parts.append(f"## Your Accumulated Findings\n{memos_block}")

        if tool_results:
            result_blocks: list[str] = []
            for tr in tool_results:
                args_brief = ", ".join(f"{k}={repr(v)[:50]}" for k, v in tr.args.items())
                status = "OK" if tr.succeeded else "FAILED"
                trunc = " [truncated]" if tr.truncated else ""
                result_blocks.append(f"### {tr.tool}({args_brief}) [{status}]{trunc}\n{tr.output}")
            parts.append("## Latest Tool Results\n" + "\n\n".join(result_blocks))

        if turns_remaining == 0:
            parts.append(
                "**Final turn.** Emit a <final> block now with all 12 template families. "
                "Mark evidence gaps explicitly rather than fabricating."
            )
        elif turns_remaining == 1:
            parts.append(
                f"You have {turns_remaining} turn remaining after this one. "
                "Consider emitting <final> soon."
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_turn(self, raw: str) -> _TurnParsed:
        think = _extract_xml_block(raw, "think")
        memo = _extract_xml_block(raw, "memo")
        tools_text = _extract_xml_block(raw, "tools")
        final_text = _extract_xml_block(raw, "final")

        tool_calls: list[_ToolCall] = []
        if tools_text:
            for line in tools_text.strip().splitlines():
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj: dict[str, Any] = json.loads(line)
                    name = str(obj.pop("tool", "")).strip()
                    if name:
                        tool_calls.append(_ToolCall(tool=name, args=obj))
                except json.JSONDecodeError:
                    pass
            tool_calls = tool_calls[: self.MAX_TOOL_CALLS_PER_TURN]

        return _TurnParsed(
            think=think,
            memo=memo,
            tool_calls=tool_calls,
            final_output=final_text,
            is_final=bool(final_text.strip()),
        )

    def _parse_final_output(self, text: str) -> dict[str, tuple[str, str]]:
        json_text = text.strip()
        fence = _JSON_FENCE_RE.search(json_text)
        if fence:
            json_text = fence.group(1).strip()
        try:
            parsed: dict[str, Any] = json.loads(json_text)
        except json.JSONDecodeError:
            return {}
        result: dict[str, tuple[str, str]] = {}
        for family, entry in parsed.items():
            if not isinstance(entry, dict):
                continue
            q = str(entry.get("question", "")).strip()
            a = str(entry.get("answer", "")).strip()
            if q and a:
                result[family] = (q, a)
        return result

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool_calls(
        self,
        calls: list[_ToolCall],
        section_id: str,
        turn_idx: int,
    ) -> list[_ToolResult]:
        results: list[_ToolResult] = []
        for call in calls:
            tr = self._execute_one(call)
            _LOG.debug(
                "section=%s turn=%d tool=%s succeeded=%s chars=%d",
                section_id,
                turn_idx,
                call.tool,
                tr.succeeded,
                len(tr.output),
            )
            results.append(tr)
        return results

    def _execute_one(self, call: _ToolCall) -> _ToolResult:
        try:
            if call.tool == "search":
                raw_out = self._tool_search(call.args)
            elif call.tool == "grep_context":
                raw_out = self._tool_grep_context(call.args)
            elif call.tool == "head":
                raw_out = self._tool_head(call.args)
            elif call.tool == "find":
                raw_out = self._tool_find(call.args)
            else:
                return _ToolResult(
                    tool=call.tool,
                    args=call.args,
                    output=f"Unknown tool '{call.tool}'. Available: search, grep_context, head, find.",
                    succeeded=False,
                    truncated=False,
                )
            output, truncated = self._truncate(raw_out)
            return _ToolResult(
                tool=call.tool, args=call.args, output=output, succeeded=True, truncated=truncated
            )
        except Exception as exc:
            return _ToolResult(
                tool=call.tool,
                args=call.args,
                output=f"Execution error: {exc}",
                succeeded=False,
                truncated=False,
            )

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
        path = _normalize_rel_path(str(args.get("path", "")))
        if not path:
            return "Error: 'path' is required."
        lines = max(10, min(400, int(args.get("lines", 200))))
        # Prefer direct read (no CLI dependency); fall back to head command.
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

    # ------------------------------------------------------------------
    # Fallback: force final synthesis from accumulated memos
    # ------------------------------------------------------------------

    def _force_final(
        self,
        section: CurriculumSection,
        memo_history: list[str],
    ) -> dict[str, tuple[str, str]]:
        """One unconditional LLM call demanding a ``<final>`` block.

        Uses the accumulated memo history as evidence; seeds from the section
        IR when no memos were written.
        """
        memos_text = (
            "\n\n".join(f"[memo/{i + 1}]\n{m}" for i, m in enumerate(memo_history))
            if memo_history
            else "(no memos written — use seed evidence only)"
        )
        symbols_line = ", ".join(list(section.core_symbols)[:12])
        contracts_block = "\n".join(list(section.api_contracts)[:8])
        prompt = (
            f"## Section: {section.section_id}\n"
            f"Objective: {section.section_objective}\n\n"
            f"## Accumulated Findings\n{memos_text}\n\n"
            "## Seed Evidence\n"
            f"Symbols: {symbols_line}\n"
            f"Contracts:\n{contracts_block}\n\n"
            "You MUST output a <final> block now with all 12 template families. "
            "Mark evidence gaps explicitly rather than fabricating."
        )
        try:
            raw = self.llm_client.generate(self._SYSTEM_PROMPT, prompt, max_tokens=3500)
            final_text = _extract_xml_block(raw, "final")
            if not final_text:
                # Some models skip the tag and return bare JSON — accept it.
                final_text = raw.strip()
            return self._parse_final_output(final_text)
        except Exception as exc:
            _LOG.warning("_force_final failed for section=%s: %s", section.section_id, exc)
            return {}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.MAX_TOOL_OUTPUT_CHARS:
            return text, False
        return text[: self.MAX_TOOL_OUTPUT_CHARS] + "\n…[output truncated]", True

    def _safe_repo_file(self, rel_path: str) -> Path | None:
        """Return an absolute path only if it resolves safely inside *repo_root*."""
        candidate = (self.repo_root / rel_path).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError:
            return None
        return candidate


class TeacherAgent:
    """Compiles source-grounded curriculum sections, units, and examples."""

    def __init__(
        self,
        paths: PipelinePaths,
        retriever: "HybridRetriever | ChromaRetriever",
        llm_client: "LLMClient | None" = None,
        cli_executor: "CliExecutor | None" = None,
        max_section_iterations: int = 5,
        saturation_patience: int = 2,
    ) -> None:
        self.paths = paths
        self.retriever = retriever
        self.llm_client = llm_client
        self.cli_executor = cli_executor
        self.max_section_iterations = max(1, max_section_iterations)
        self.saturation_patience = max(1, saturation_patience)

    def run(self, resume: bool = True) -> dict[str, int]:
        self.paths.dataset_dir.mkdir(parents=True, exist_ok=True)
        agenda = self._parse_investigator_records(load_findings(self.paths.investigator_dir))

        section_partial_path = self.paths.dataset_dir / "curriculum_sections.partial.jsonl"
        unit_partial_path = self.paths.dataset_dir / "topic_units.partial.jsonl"
        dataset_partial_path = self.paths.dataset_dir / "instruction_dataset.partial.jsonl"
        dataset_final_path = self.paths.dataset_dir / "instruction_dataset.jsonl"

        if not resume:
            for path in (section_partial_path, unit_partial_path, dataset_partial_path):
                if path.exists():
                    path.unlink()

        sections = self._compile_sections(agenda, section_partial_path, resume=resume)
        units = self._compile_units(sections, unit_partial_path, resume=resume)
        dataset_rows = self._compile_examples(sections, units, dataset_partial_path, resume=resume)

        if dataset_partial_path.exists():
            dataset_partial_path.replace(dataset_final_path)
        else:
            dataset_final_path.write_text("", encoding="utf-8")

        curriculum_payload = {
            "schema": "distillme.curriculum.v1",
            "sections": [section.to_jsonable() for section in sections],
            "section_order": [spec.section_id for spec in _SECTION_SPECS],
            "prerequisite_graph": [
                {
                    "section_id": section.section_id,
                    "prerequisite_sections": list(section.prerequisite_sections),
                }
                for section in sections
            ],
        }
        (self.paths.dataset_dir / "curriculum.json").write_text(
            json.dumps(curriculum_payload, indent=2), encoding="utf-8"
        )

        topic_units_payload = {
            "schema": "distillme.topic_units.v1",
            "units": [unit.to_jsonable() for unit in units],
            "unit_count": len(units),
        }
        (self.paths.dataset_dir / "topic_units.json").write_text(
            json.dumps(topic_units_payload, indent=2), encoding="utf-8"
        )

        coverage_map = self._build_coverage_map(sections, units, dataset_rows)
        (self.paths.dataset_dir / "coverage_map.json").write_text(
            json.dumps(coverage_map, indent=2), encoding="utf-8"
        )

        teacher_report = self._build_teacher_report(agenda, sections, units, dataset_rows)
        (self.paths.dataset_dir / "teacher_report.json").write_text(
            json.dumps(teacher_report, indent=2), encoding="utf-8"
        )

        categories = sorted({str(row.get("task_category", "")) for row in dataset_rows if row.get("task_category")})
        difficulties = sorted({str(row.get("difficulty", "")) for row in dataset_rows if row.get("difficulty")})
        template_families = sorted({str(row.get("template_family", "")) for row in dataset_rows if row.get("template_family")})
        manifest = {
            "schema": "distillme.dataset.v2",
            "examples": len(dataset_rows),
            "sections": len(sections),
            "units": len(units),
            "categories": categories,
            "difficulties": difficulties,
            "template_families": template_families,
            "quality_controls": [
                "source_reopen_validation_for_retrieval_hits",
                "section_and_unit_traceability",
                "high_confidence_evidence_density_guardrail",
                "contrastive_claim_contradiction_check",
                "difficulty_score_consistency",
                "coverage_balance_monitoring",
            ],
        }
        (self.paths.dataset_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        return {
            "sections": len(sections),
            "units": len(units),
            "examples": len(dataset_rows),
            "investigator_records": len(agenda),
        }

    def _compile_sections(
        self,
        agenda: list[InvestigatorFindingRecord],
        partial_path: Path,
        resume: bool,
    ) -> list[CurriculumSection]:
        loaded: dict[str, dict[str, Any]] = {}
        if resume and partial_path.exists():
            loaded = _load_partial_map(partial_path, "section_id")

        sections: list[CurriculumSection] = []
        for spec in _SECTION_SPECS:
            if spec.section_id in loaded:
                sections.append(self._section_from_json(loaded[spec.section_id]))
                continue
            section = self._compile_one_section(spec, agenda)
            _append_jsonl(partial_path, section.to_jsonable())
            sections.append(section)
        return sections

    def _compile_units(
        self,
        sections: list[CurriculumSection],
        partial_path: Path,
        resume: bool,
    ) -> list[TopicUnit]:
        loaded: dict[str, dict[str, Any]] = {}
        if resume and partial_path.exists():
            loaded = _load_partial_map(partial_path, "unit_id")

        units: list[TopicUnit] = []
        for section in sections:
            for unit in self._build_units_for_section(section):
                if unit.unit_id in loaded:
                    units.append(self._topic_unit_from_json(loaded[unit.unit_id]))
                    continue
                _append_jsonl(partial_path, unit.to_jsonable())
                units.append(unit)
        units.sort(key=lambda item: item.unit_id)
        return units

    def _compile_examples(
        self,
        sections: list[CurriculumSection],
        units: list[TopicUnit],
        partial_path: Path,
        resume: bool,
    ) -> list[dict[str, Any]]:
        completed_ids: set[str] = set()
        if resume and partial_path.exists():
            for row in _load_jsonl(partial_path):
                task_id = str(row.get("task_id", "")).strip()
                if task_id:
                    completed_ids.add(task_id)

        sections_by_id = {section.section_id: section for section in sections}

        # Pre-compute LLM-synthesized Q&A for each section (one call per section).
        # Units that are already completed are skipped; sections with no pending
        # units are not sent to the LLM.
        pending_section_ids = {
            sections_by_id[unit.section_id].section_id
            for unit in units
            if self._task_id_for_unit(unit.unit_id) not in completed_ids
        }
        synthesis_cache: dict[str, dict[str, tuple[str, str]]] = {}
        for section in sections:
            if section.section_id in pending_section_ids:
                synthesis_cache[section.section_id] = self._synthesize_section_qa_batch(section)

        with partial_path.open("a", encoding="utf-8") as handle:
            for unit in units:
                task_id = self._task_id_for_unit(unit.unit_id)
                if task_id in completed_ids:
                    continue
                section = sections_by_id[unit.section_id]
                synthesized_qa = synthesis_cache.get(section.section_id, {})
                example = self._example_from_unit(section, unit, synthesized_qa)
                handle.write(json.dumps(example.to_jsonable(), sort_keys=True) + "\n")
                handle.flush()
                completed_ids.add(task_id)

        return _load_jsonl(partial_path)

    def _parse_investigator_records(self, findings: list[dict[str, str]]) -> list[InvestigatorFindingRecord]:
        records: list[InvestigatorFindingRecord] = []
        for item in findings:
            path = str(item.get("path", ""))
            text = str(item.get("text", ""))
            topic = path.removesuffix(".md")
            title = self._extract_scalar(text, "- title:", default=topic.replace("_", " ").title())
            confidence_raw = self._extract_scalar(text, "- confidence:", default="0.0")
            try:
                confidence = float(confidence_raw)
            except ValueError:
                confidence = 0.0
            record = InvestigatorFindingRecord(
                document_name=path,
                topic=topic,
                finding_title=title,
                confidence=confidence,
                supporting_evidence_references=tuple(self._extract_bullets(text, "- supporting evidence:")),
                source_file_references=tuple(self._extract_bullets(text, "- source file references:")),
                unresolved_ambiguity=self._extract_scalar(text, "- unresolved ambiguity:", default=""),
                architectural_implications=self._extract_scalar(text, "- architectural implications:", default=""),
                investigation_iterations=tuple(self._parse_iteration_blocks(text)),
                commands_executed=tuple(self._extract_bullets(text, "**COMMANDS RUN:**")),
                findings_discovered_during_trace=tuple(self._extract_bullets(text, "**KNOWN EVIDENCE:**")),
                model_analysis=self._extract_section(
                    text,
                    start_heading="## Model Analysis",
                    end_headings=("## Unresolved Questions",),
                ),
            )
            records.append(record)
        return records

    def _parse_iteration_blocks(self, text: str) -> list[dict[str, Any]]:
        lines = text.splitlines()
        parsed: list[dict[str, Any]] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.startswith("#### Iteration "):
                index += 1
                continue
            iteration_label = line.removeprefix("#### Iteration ").strip()
            try:
                iteration_num = max(0, int(iteration_label) - 1)
            except ValueError:
                iteration_num = len(parsed)
            block: list[str] = []
            index += 1
            while index < len(lines):
                current = lines[index]
                if current.startswith("#### Iteration "):
                    break
                if current.strip() == "**KNOWN EVIDENCE:**":
                    break
                if current.startswith("## "):
                    break
                block.append(current)
                index += 1
            parsed.append(
                {
                    "iteration_num": iteration_num,
                    "plan": self._extract_scalar_from_lines(block, "**Plan:**"),
                    "commands": self._extract_bullets_from_lines(block, "**Commands:**"),
                    "findings": self._extract_bullets_from_lines(block, "**Findings:**"),
                    "hypothesis_after": self._extract_scalar_from_lines(block, "**Hypothesis after:**"),
                }
            )
        return parsed

    def _compile_one_section(
        self,
        spec: _SectionSpec,
        agenda: list[InvestigatorFindingRecord],
    ) -> CurriculumSection:
        relevant = [record for record in agenda if record.topic in spec.agenda_topics]
        if not relevant:
            relevant = list(agenda)

        acc = _SectionAccumulator(
            core_files=set(),
            core_symbols=set(),
            api_contracts=set(),
            recurring_conventions=set(),
            best_practices=set(),
            invariants=set(),
            execution_flows=set(),
            failure_modes=set(),
            evidence_refs=set(),
            ambiguities=set(),
            commands_executed=set(),
            retrieved_contexts=[],
            investigator_topics=set(),
            iterations=0,
        )

        self._seed_from_investigator_records(relevant, acc)

        stale_rounds = 0
        for iteration in range(self.max_section_iterations):
            top_k = self._adaptive_top_k(iteration, len(acc.ambiguities))
            query = self._build_section_query(spec, acc)
            hits = self.retriever.search(query, top_k=top_k)
            added = self._consume_retrieval_hits(hits, acc)
            added += self._run_cli_expansion(spec, acc, iteration)
            acc.iterations = iteration + 1
            if added == 0:
                stale_rounds += 1
            else:
                stale_rounds = 0
            if stale_rounds >= self.saturation_patience:
                break

        # Broad fallback: if no files found by section-specific seed, use a generic
        # repository-wide query so that every section has at least one representative file.
        if not acc.core_files:
            broad_hits = self.retriever.search(spec.section_objective, top_k=4)
            if broad_hits:
                self._consume_retrieval_hits(broad_hits, acc)
            # Last resort: take any indexed files via an empty-style query.
            if not acc.core_files:
                any_hits = self.retriever.search("class method function", top_k=4)
                if any_hits:
                    self._consume_retrieval_hits(any_hits, acc)

        if not acc.evidence_refs:
            acc.ambiguities.add(
                "No source evidence could be validated for this section; defer high-confidence claims."
            )

        if not acc.recurring_conventions:
            acc.recurring_conventions.add(
                "Repository-specific claims are only accepted after reopening source files directly."
            )
        if not acc.best_practices:
            acc.best_practices.add(
                "Prefer targeted retrieval + CLI expansion loops over one-shot broad summarization."
            )

        confidence = self._confidence_score(acc)
        difficulty = self._difficulty_score(spec, acc)

        return CurriculumSection(
            section_id=spec.section_id,
            section_objective=spec.section_objective,
            prerequisite_sections=spec.prerequisite_sections,
            core_files=tuple(_limit_sorted(acc.core_files, 80)),
            core_symbols=tuple(_limit_sorted(acc.core_symbols, 120)),
            api_contracts=tuple(_limit_sorted(acc.api_contracts, 120)),
            recurring_codebase_conventions=tuple(_limit_sorted(acc.recurring_conventions, 30)),
            best_practice_patterns=tuple(_limit_sorted(acc.best_practices, 30)),
            invariants=tuple(_limit_sorted(acc.invariants, 80)),
            execution_flows=tuple(_limit_sorted(acc.execution_flows, 120)),
            failure_modes=tuple(_limit_sorted(acc.failure_modes, 120)),
            evidence_references=tuple(_limit_sorted(acc.evidence_refs, 200)),
            confidence_score=confidence,
            ambiguity_open_questions=tuple(_limit_sorted(acc.ambiguities, 40)),
            recommended_task_families=spec.recommended_task_families,
            difficulty_score=difficulty,
            investigator_topics=tuple(_limit_sorted(acc.investigator_topics, 40)),
            compilation_iterations=acc.iterations,
        )

    def _seed_from_investigator_records(
        self,
        records: list[InvestigatorFindingRecord],
        acc: _SectionAccumulator,
    ) -> None:
        for record in records:
            acc.investigator_topics.add(record.topic)
            if record.unresolved_ambiguity:
                acc.ambiguities.add(record.unresolved_ambiguity)
            if record.architectural_implications:
                acc.recurring_conventions.add(record.architectural_implications)
            for command in record.commands_executed:
                _add_capped(acc.commands_executed, command, 160)
            for evidence in record.supporting_evidence_references + record.source_file_references:
                parsed = _parse_source_ref(evidence)
                if parsed is not None:
                    path, start_line, end_line = parsed
                    if self._ingest_source_range(path, start_line, end_line, acc, reason="investigator_seed"):
                        _add_capped(acc.evidence_refs, f"{path}:{start_line}-{end_line}", 260)
                else:
                    for path in _extract_paths_from_text(evidence):
                        self._ingest_source_range(path, 1, 160, acc, reason="investigator_seed")
                        _add_capped(acc.evidence_refs, f"{path}:1-160", 260)
            for finding in record.findings_discovered_during_trace:
                for path in _extract_paths_from_text(finding):
                    self._ingest_source_range(path, 1, 160, acc, reason="trace_seed")
                    _add_capped(acc.evidence_refs, f"{path}:1-160", 260)

    def _adaptive_top_k(self, iteration: int, ambiguity_count: int) -> int:
        if iteration <= 0:
            base = 8
        elif iteration == 1:
            base = 12
        elif iteration == 2:
            base = 16
        else:
            base = 24
        if ambiguity_count >= 4:
            base = min(24, base + 4)
        return max(8, min(24, base))

    def _build_section_query(self, spec: _SectionSpec, acc: _SectionAccumulator) -> str:
        parts = [spec.section_objective, " ".join(spec.seed_terms)]
        if acc.core_symbols:
            parts.append(" ".join(_limit_sorted(acc.core_symbols, 8)))
        if acc.core_files:
            parts.append(" ".join(Path(path).name for path in _limit_sorted(acc.core_files, 8)))
        if acc.evidence_refs:
            parts.append(" ".join(_limit_sorted(acc.evidence_refs, 3)))
        return " ".join(part for part in parts if part.strip())

    def _consume_retrieval_hits(self, hits: list[RetrievalHit], acc: _SectionAccumulator) -> int:
        added = 0
        for hit in hits:
            chunk = hit.chunk
            path = _normalize_rel_path(chunk.path)
            if not path:
                continue
            if not self._ingest_source_range(path, chunk.start_line, chunk.end_line, acc, reason="retrieval_validation"):
                continue
            ref = f"{path}:{chunk.start_line}-{chunk.end_line}"
            if _add_capped(acc.evidence_refs, ref, 260):
                added += 1
            if _add_capped(acc.core_files, path, 120):
                added += 1
            for symbol in chunk.symbols:
                if _add_capped(acc.core_symbols, symbol, 240):
                    added += 1
        return added

    def _run_cli_expansion(self, spec: _SectionSpec, acc: _SectionAccumulator, iteration: int) -> int:
        if self.cli_executor is None:
            return 0
        added = 0
        discovered_paths: set[str] = set()
        commands = self._section_cli_commands(spec, acc, iteration)
        for args in commands:
            try:
                result = self.cli_executor.run(args)
            except ValueError:
                continue
            _add_capped(acc.commands_executed, " ".join(args), 200)
            if not result.stdout.strip():
                continue
            refs, paths = self._extract_refs_from_cli_output(args, result.stdout)
            discovered_paths.update(paths)
            for ref in refs:
                if _add_capped(acc.evidence_refs, ref, 260):
                    added += 1

        inspected = 0
        for path in sorted(discovered_paths):
            if inspected >= 6:
                break
            if self._ingest_source_range(path, 1, 180, acc, reason="cli_followup"):
                inspected += 1
                if _add_capped(acc.core_files, path, 120):
                    added += 1
        return added

    def _section_cli_commands(self, spec: _SectionSpec, acc: _SectionAccumulator, iteration: int) -> list[list[str]]:
        commands: list[list[str]] = []
        pattern_terms = list(spec.seed_terms)
        pattern_terms.extend(_limit_sorted(acc.core_symbols, 3))
        escaped_terms = [re.escape(term) for term in pattern_terms if term]
        if escaped_terms:
            context_lines = "50" if iteration == 0 else "70"
            commands.append(["grep_context", "|".join(escaped_terms[:4]), ".", context_lines])

        if iteration == 0:
            commands.extend(
                [
                    ["find", ".", "-name", "*test*.py", "-type", "f"],
                    ["find", ".", "-name", "*config*.py", "-type", "f"],
                    ["find", ".", "-name", "pyproject.toml", "-type", "f"],
                    ["find", ".", "-name", "*.json", "-type", "f"],
                ]
            )
        else:
            if acc.core_symbols:
                symbol_pattern = "|".join(re.escape(item) for item in _limit_sorted(acc.core_symbols, 3))
                commands.append(["grep_context", symbol_pattern, ".", "60"])
            commands.extend(
                [
                    ["find", ".", "-name", "*test*.py", "-type", "f"],
                    ["find", ".", "-name", "*.toml", "-type", "f"],
                    ["find", ".", "-name", "*.yaml", "-type", "f"],
                    ["find", ".", "-name", "*.yml", "-type", "f"],
                ]
            )
        return commands

    def _extract_refs_from_cli_output(self, args: list[str], stdout: str) -> tuple[list[str], set[str]]:
        refs: list[str] = []
        discovered_paths: set[str] = set()
        cmd = args[0]

        if cmd == "find":
            for line in stdout.splitlines():
                path = _normalize_rel_path(line.strip())
                if not path:
                    continue
                discovered_paths.add(path)
                refs.append(f"{path}:1-160")
            return refs, discovered_paths

        if cmd == "grep_context":
            for line in stdout.splitlines():
                match = _GREP_CONTEXT_HEADER_RE.match(line.strip())
                if not match:
                    continue
                path = _normalize_rel_path(match.group(1))
                start = int(match.group(2))
                end = start + 70
                if path:
                    discovered_paths.add(path)
                    refs.append(f"{path}:{start}-{end}")
            return refs, discovered_paths

        if cmd == "grep":
            for line in stdout.splitlines():
                if not line.strip():
                    continue
                parts = line.split(":", 2)
                if len(parts) < 2:
                    continue
                path = _normalize_rel_path(parts[0])
                if not path:
                    continue
                discovered_paths.add(path)
                try:
                    start = int(parts[1])
                except ValueError:
                    start = 1
                refs.append(f"{path}:{start}-{start + 50}")
            return refs, discovered_paths

        for line in stdout.splitlines():
            for path in _extract_paths_from_text(line):
                discovered_paths.add(path)
                refs.append(f"{path}:1-160")
        return refs, discovered_paths

    def _ingest_source_range(
        self,
        rel_path: str,
        start_line: int,
        end_line: int,
        acc: _SectionAccumulator,
        reason: str,
    ) -> bool:
        context = self._read_context_from_source(rel_path, start_line, end_line, reason)
        if context is None:
            context = self._read_context_via_head(rel_path, max(120, end_line - start_line + 40), reason)
        if context is None:
            return False

        _add_capped(acc.core_files, str(context["path"]), 120)
        for symbol in context.get("symbols", []):
            _add_capped(acc.core_symbols, str(symbol), 240)
        if len(acc.retrieved_contexts) < 60:
            acc.retrieved_contexts.append(context)
        self._digest_context_text(str(context["path"]), int(context["start_line"]), str(context["text"]), acc)
        return True

    def _read_context_from_source(
        self,
        rel_path: str,
        start_line: int,
        end_line: int,
        reason: str,
    ) -> dict[str, object] | None:
        safe_path = self._safe_repo_file(rel_path)
        if safe_path is None or not safe_path.exists() or not safe_path.is_file():
            return None
        try:
            content = safe_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        lines = content.splitlines()
        if not lines:
            return None
        start = max(1, start_line)
        end = min(len(lines), max(start, end_line))
        window_start = max(1, start - 10)
        window_end = min(len(lines), end + 20)
        text = "\n".join(lines[window_start - 1 : window_end])
        rel = safe_path.relative_to(self.paths.repository.resolve()).as_posix()
        return {
            "path": rel,
            "start_line": window_start,
            "end_line": window_end,
            "symbols": _extract_symbols(text),
            "score": 1.0,
            "text": text,
            "reasons": [reason, "source_reopened"],
        }

    def _read_context_via_head(
        self,
        rel_path: str,
        lines_to_read: int,
        reason: str,
    ) -> dict[str, object] | None:
        if self.cli_executor is None:
            return None
        path = _normalize_rel_path(rel_path)
        if not path:
            return None
        try:
            result = self.cli_executor.run(["head", f"-{max(40, lines_to_read)}", path])
        except ValueError:
            return None
        if not result.succeeded or not result.stdout.strip():
            return None
        text = result.stdout
        return {
            "path": path,
            "start_line": 1,
            "end_line": max(1, len(text.splitlines())),
            "symbols": _extract_symbols(text),
            "score": 1.0,
            "text": text,
            "reasons": [reason, "head_reopened"],
        }

    def _digest_context_text(
        self,
        path: str,
        start_line: int,
        text: str,
        acc: _SectionAccumulator,
    ) -> None:
        for offset, raw_line in enumerate(text.splitlines()):
            line_no = start_line + offset
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()

            for _, symbol in _SYMBOL_DEF_RE.findall(line):
                _add_capped(acc.core_symbols, symbol, 240)

            if "(" in line and ")" in line and (
                line.startswith("def ")
                or " class " in f" {line} "
                or re.search(r"\b(public|private|protected|async)\b", line)
            ):
                _add_capped(acc.api_contracts, f"{path}:{line_no} {line[:180]}", 160)

            if any(token in lower for token in ("raise ", "throw ", "except", "catch", "timeout", "failed", "error")):
                _add_capped(acc.failure_modes, f"{path}:{line_no} {line[:180]}", 160)

            if (
                any(token in lower for token in ("assert", "required", "must", "invalid", "null", "none"))
                and any(token in lower for token in ("if ", "raise", "throw", "return"))
            ):
                _add_capped(acc.invariants, f"{path}:{line_no} {line[:180]}", 160)

            if any(token in lower for token in ("dispatch", "handler", "run(", "execute(", "flow", "pipeline", "event")):
                _add_capped(acc.execution_flows, f"{path}:{line_no} {line[:180]}", 180)

            if (
                ("test" in path.lower() or "/tests/" in path)
                and any(token in lower for token in ("assert", "pytest", "unittest", "mock", "fixture", "parametrize"))
            ):
                _add_capped(
                    acc.recurring_conventions,
                    "Tests validate observable behavior and explicitly enumerate edge cases.",
                    40,
                )

            if path.endswith(".py") and line.startswith("def ") and "->" in line:
                _add_capped(
                    acc.best_practices,
                    "Prefer explicit type-annotated function signatures in Python modules.",
                    40,
                )

            if (
                path.endswith("pyproject.toml")
                or path.endswith(".json")
                or path.endswith(".yaml")
                or path.endswith(".yml")
            ):
                if "=" in line or ":" in line:
                    _add_capped(
                        acc.recurring_conventions,
                        "Configuration values are declared in dedicated manifest/config files.",
                        40,
                    )

    def _confidence_score(self, acc: _SectionAccumulator) -> float:
        evidence_density = min(1.0, len(acc.evidence_refs) / 16.0)
        symbol_density = min(1.0, len(acc.core_symbols) / 30.0)
        ambiguity_penalty = min(0.35, 0.05 * len(acc.ambiguities))
        value = 0.30 + 0.45 * evidence_density + 0.20 * symbol_density - ambiguity_penalty
        return round(max(0.05, min(0.95, value)), 3)

    def _difficulty_score(self, spec: _SectionSpec, acc: _SectionAccumulator) -> float:
        modules = {path.split("/", 1)[0] for path in acc.core_files if path}
        cross_module_span = len(modules)
        inheritance_depth = sum(
            1
            for item in acc.api_contracts
            if " extends " in item.lower() or " implements " in item.lower()
        )
        call_chain_length = min(6, len(acc.execution_flows))
        ambiguity = len(acc.ambiguities)
        tool_complexity = min(6, len(acc.commands_executed) // 3)
        sensitivity = 1 if any(token in spec.section_id for token in ("security", "concurrency", "performance")) else 0
        has_tests = any("test" in path.lower() for path in acc.core_files)
        has_prod = any("test" not in path.lower() for path in acc.core_files)
        test_prod_coupling = 1 if has_tests and has_prod else 0

        score = 1.0
        score += min(2.0, cross_module_span * 0.35)
        score += min(1.2, inheritance_depth * 0.2)
        score += min(1.4, call_chain_length * 0.2)
        score += min(1.5, ambiguity * 0.25)
        score += min(1.2, tool_complexity * 0.2)
        score += 1.0 * sensitivity
        score += 0.8 * test_prod_coupling
        return round(max(1.0, min(10.0, score)), 2)

    def _build_units_for_section(self, section: CurriculumSection) -> list[TopicUnit]:
        units: list[TopicUnit] = []
        counters: defaultdict[str, int] = defaultdict(int)

        def _next_unit_id(template_family: str) -> str:
            counters[template_family] += 1
            return self._unit_id(section.section_id, template_family, counters[template_family])

        section_concept_id = self._concept_unit_id(section.section_id)
        prerequisite_concepts = tuple(self._concept_unit_id(item) for item in section.prerequisite_sections)
        base_prereqs = tuple(dict.fromkeys((*prerequisite_concepts, section_concept_id)))

        supporting_files = tuple(_limit_sorted(set(section.core_files), 8))
        symbols = tuple(_limit_sorted(set(section.core_symbols), 10))
        evidence_refs = tuple(_limit_sorted(set(section.evidence_references), 16))

        concept_payload = {
            "objective": section.section_objective,
            "key_concepts": list(symbols[:6]) or [section.section_id],
            "supporting_evidence": list(evidence_refs[:8]),
            "concise_explanation": (
                f"{section.section_id.replace('_', ' ')} is grounded by reopened source files and "
                "iterative retrieval + CLI expansion passes."
            ),
            "uncertainty_notes": list(section.ambiguity_open_questions[:3]),
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("concept_card"),
                section_id=section.section_id,
                objective=section.section_objective,
                template_family="concept_card",
                supervision_track="concept",
                prerequisite_unit_ids=prerequisite_concepts,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "concept_card"),
                confidence_score=self._unit_confidence(section.confidence_score, "concept_card"),
                payload=concept_payload,
            )
        )

        contracts = list(section.api_contracts[:2])
        if not contracts:
            fallback_symbol = symbols[0] if symbols else section.section_id
            contracts = [f"{fallback_symbol}: contract requires source-level verification."]
        for contract in contracts:
            symbol = _extract_symbol_from_contract(contract)
            payload = {
                "symbol": symbol,
                "contract_summary": contract,
                "inputs": ["Inputs must follow observed source signatures."],
                "outputs": ["Outputs are constrained by reopened implementation and tests."],
                "invariants": list(section.invariants[:3]),
                "failure_behavior": list(section.failure_modes[:3]),
                "evidence_refs": list(evidence_refs[:8]),
            }
            units.append(
                TopicUnit(
                    unit_id=_next_unit_id("api_contract_card"),
                    section_id=section.section_id,
                    objective=f"Contract extraction for {symbol}",
                    template_family="api_contract_card",
                    supervision_track="concept",
                    prerequisite_unit_ids=base_prereqs,
                    evidence_references=evidence_refs,
                    supporting_files=supporting_files,
                    symbols=symbols,
                    difficulty_score=self._unit_difficulty(section.difficulty_score, "api_contract_card"),
                    confidence_score=self._unit_confidence(section.confidence_score, "api_contract_card"),
                    payload=payload,
                )
            )

        flow_hops = list(section.execution_flows[:6])
        if not flow_hops:
            flow_hops = [f"Inspect flow from {path}" for path in supporting_files[:3]]
        flow_payload = {
            "entry_point": symbols[0] if symbols else (supporting_files[0] if supporting_files else section.section_id),
            "ordered_hops": flow_hops,
            "transition_rationale": "Hops were accepted only after direct source reopen validation.",
            "terminal_effect": "Flow reaches a repository-observed state transition or output side effect.",
            "uncertainty_points": list(section.ambiguity_open_questions[:2]),
            "evidence_refs": list(evidence_refs[:10]),
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("execution_flow_trace"),
                section_id=section.section_id,
                objective="Execution flow reconstruction",
                template_family="execution_flow_trace",
                supervision_track="concept",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "execution_flow_trace"),
                confidence_score=self._unit_confidence(section.confidence_score, "execution_flow_trace"),
                payload=flow_payload,
            )
        )

        retrieval_payload = {
            "task": f"Navigate section '{section.section_id}' and ground one implementation decision.",
            "initial_clues": list(evidence_refs[:4]),
            "tool_sequence": [
                "retriever.search(query, top_k=8) for focused symbol lookup",
                "grep_context <symbol_or_pattern> . 60",
                "head -200 <candidate_file>",
                "find . -name '*test*.py' -type f",
                "retriever.search(query, top_k=12-16) for medium synthesis",
                "retriever.search(query, top_k=24) for broad ambiguous sweep",
            ],
            "expected_observations": [
                "Primary symbol definitions and direct callers.",
                "Related tests, config, or build declarations.",
                "Any unresolved ambiguity that requires deferral.",
            ],
            "stopping_condition": (
                "Stop when consecutive passes no longer reveal materially new contracts, flows, or constraints."
            ),
            "final_grounded_answer": "Answer cites reopened file locations and explicitly marks uncertainty.",
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("retrieval_and_tool_use_task"),
                section_id=section.section_id,
                objective="Agentic navigation and retrieval workflow",
                template_family="retrieval_and_tool_use_task",
                supervision_track="action",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "retrieval_and_tool_use_task"),
                confidence_score=self._unit_confidence(section.confidence_score, "retrieval_and_tool_use_task"),
                payload=retrieval_payload,
            )
        )

        implementation_payload = {
            "change_goal": f"Implement a repository-consistent change in section '{section.section_id}'.",
            "impacted_files": list(supporting_files[:6]),
            "constraints": list(section.invariants[:4])
            or ["Maintain observed API contracts and preserve uncertainty handling."],
            "ordered_steps": [
                "Locate and reopen the authoritative symbols for the change.",
                "Draft modifications respecting existing contracts and conventions.",
                "Update or add tests that verify observable behavior.",
                "Re-run retrieval/CLI checks to catch missed call sites.",
            ],
            "validations": [
                "Supporting files exist and were reopened.",
                "Failure modes remain explicitly handled.",
                "Question/answer pair cites concrete evidence refs.",
            ],
            "risks": list(section.failure_modes[:3])
            or ["Cross-module side effects may exist outside currently indexed scope."],
            "evidence_refs": list(evidence_refs[:10]),
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("implementation_plan_task"),
                section_id=section.section_id,
                objective="Implementation and refactoring planning",
                template_family="implementation_plan_task",
                supervision_track="action",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "implementation_plan_task"),
                confidence_score=self._unit_confidence(section.confidence_score, "implementation_plan_task"),
                payload=implementation_payload,
            )
        )

        debugging_payload = {
            "observed_symptom": section.failure_modes[0] if section.failure_modes else "Unexpected behavior in a source-validated path.",
            "likely_root_cause": (
                "Guardrails or preconditions are missing in one reopened branch."
                if section.failure_modes
                else "Insufficient explicit checks in the observed code path."
            ),
            "evidence_path": list(evidence_refs[:6]),
            "minimal_fix": "Add the smallest precondition check that restores the observed contract.",
            "validation_strategy": "Run affected tests and verify no contract regression in callers.",
            "uncertainty_notes": list(section.ambiguity_open_questions[:2]),
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("debugging_and_failure_analysis_task"),
                section_id=section.section_id,
                objective="Debugging and failure analysis",
                template_family="debugging_and_failure_analysis_task",
                supervision_track="action",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "debugging_and_failure_analysis_task"),
                confidence_score=self._unit_confidence(section.confidence_score, "debugging_and_failure_analysis_task"),
                payload=debugging_payload,
            )
        )

        has_tests = any("test" in path.lower() for path in supporting_files)
        test_payload = {
            "target_behavior": f"Behavioral contract for section '{section.section_id}'.",
            "setup": "Use fixtures that mirror reopened production entry points.",
            "critical_cases": [
                "Happy path from observed execution flow.",
                "Failure path derived from exception or guardrail evidence.",
                "Ambiguous path that should produce a cautious deferral response.",
            ],
            "assertions": [
                "Assert observable outputs and side effects.",
                "Assert explicit uncertainty language when evidence is insufficient.",
            ],
            "mocks_dependencies": [
                "Mock external systems not represented by local indexed artifacts.",
            ],
            "evidence_refs": list(evidence_refs[:8]),
            "has_observed_tests": has_tests,
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("test_design_task"),
                section_id=section.section_id,
                objective="Test design and validation pattern transfer",
                template_family="test_design_task",
                supervision_track="action",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "test_design_task"),
                confidence_score=self._unit_confidence(section.confidence_score, "test_design_task"),
                payload=test_payload,
            )
        )

        tempting_claim = (
            f"{symbols[0] if symbols else section.section_id} guarantees behavior across all runtime paths."
        )
        contrastive_payload = {
            "tempting_but_unsupported_claim": tempting_claim,
            "why_unsupported_or_wrong": (
                "The reopened evidence does not cover all runtime paths; broad guarantee is unsupported."
            ),
            "correcting_evidence": list(evidence_refs[:6]),
            "proper_cautious_answer": (
                "Limit claims to reopened files and explicitly defer unresolved runtime behavior."
            ),
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("contrastive_negative_example"),
                section_id=section.section_id,
                objective="Contrastive hallucination-resistance training",
                template_family="contrastive_negative_example",
                supervision_track="uncertainty",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "contrastive_negative_example"),
                confidence_score=self._unit_confidence(section.confidence_score, "contrastive_negative_example"),
                payload=contrastive_payload,
            )
        )

        ambiguity_payload = {
            "question": section.ambiguity_open_questions[0]
            if section.ambiguity_open_questions
            else f"What is still uncertain in section '{section.section_id}'?",
            "available_evidence": list(evidence_refs[:6]),
            "ambiguity_source": (
                section.ambiguity_open_questions[0]
                if section.ambiguity_open_questions
                else "Current static evidence does not prove dynamic behavior."
            ),
            "safe_answer": "State what is known, what is unknown, and why deferral is required.",
            "next_retrieval_step": "Run targeted grep_context + source reopen on the unresolved symbol path.",
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("ambiguity_and_deferral_example"),
                section_id=section.section_id,
                objective="Safe uncertainty and deferral behavior",
                template_family="ambiguity_and_deferral_example",
                supervision_track="uncertainty",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "ambiguity_and_deferral_example"),
                confidence_score=self._unit_confidence(section.confidence_score, "ambiguity_and_deferral_example"),
                payload=ambiguity_payload,
            )
        )

        design_payload = {
            "engineering_goal": f"Choose a repository-consistent design for section '{section.section_id}'.",
            "candidate_approaches": [
                "Extend existing abstraction with minimal API surface changes.",
                "Introduce a new helper while preserving current entry points.",
                "Refactor call path and update tests in lockstep.",
            ],
            "apis_or_extension_points_considered": list(symbols[:4]),
            "observed_codebase_precedents": list(supporting_files[:4]),
            "tradeoffs": [
                "Lower implementation churn vs. future extensibility.",
                "Local simplicity vs. cross-module reuse.",
            ],
            "chosen_approach": "Prefer the smallest change that matches existing contracts and tests.",
            "rejected_alternatives": [
                "Large rewrites unsupported by current evidence density.",
            ],
            "validation_plan": [
                "Reopen changed files and nearest callers.",
                "Verify affected tests and configuration paths.",
            ],
            "evidence_refs": list(evidence_refs[:10]),
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("design_deliberation_trace"),
                section_id=section.section_id,
                objective="Reasoning supervision for design deliberation",
                template_family="design_deliberation_trace",
                supervision_track="reasoning",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "design_deliberation_trace"),
                confidence_score=self._unit_confidence(section.confidence_score, "design_deliberation_trace"),
                payload=design_payload,
            )
        )

        best_practice_payload = {
            "practice": section.best_practice_patterns[0]
            if section.best_practice_patterns
            else "Prefer explicit evidence refs in every repository-specific answer.",
            "where_observed": list(supporting_files[:4]),
            "why_it_likely_exists": "It keeps repository-specialized answers grounded and reproducible.",
            "when_to_apply": "Any code change, API choice, or debugging recommendation.",
            "when_not_to_apply": "Do not over-generalize when evidence is sparse or contradictory.",
            "evidence_refs": list(evidence_refs[:8]),
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("best_practice_transfer_card"),
                section_id=section.section_id,
                objective="Transfer repository best-practice conventions",
                template_family="best_practice_transfer_card",
                supervision_track="reasoning",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "best_practice_transfer_card"),
                confidence_score=self._unit_confidence(section.confidence_score, "best_practice_transfer_card"),
                payload=best_practice_payload,
            )
        )

        candidate_apis = list(symbols[:3])
        if len(candidate_apis) < 2:
            candidate_apis.extend([Path(path).stem for path in supporting_files[:2]])
        candidate_apis = [item for item in candidate_apis if item]
        if not candidate_apis:
            candidate_apis = [section.section_id]
        api_selection_payload = {
            "task": f"Select the most compatible API/extension point for section '{section.section_id}'.",
            "candidate_apis": candidate_apis,
            "compatibility_constraints": list(section.invariants[:3])
            or ["Preserve current call signatures and failure handling."],
            "precedent_in_existing_code": list(supporting_files[:4]),
            "risks_of_each_option": [
                f"{api}: verify caller compatibility and tests before adoption." for api in candidate_apis
            ],
            "selected_api": candidate_apis[0],
            "rationale": "Selected option has strongest reopened evidence and smallest migration risk.",
            "evidence_refs": list(evidence_refs[:8]),
        }
        units.append(
            TopicUnit(
                unit_id=_next_unit_id("api_selection_debate"),
                section_id=section.section_id,
                objective="Reason about API/extension-point selection",
                template_family="api_selection_debate",
                supervision_track="reasoning",
                prerequisite_unit_ids=base_prereqs,
                evidence_references=evidence_refs,
                supporting_files=supporting_files,
                symbols=symbols,
                difficulty_score=self._unit_difficulty(section.difficulty_score, "api_selection_debate"),
                confidence_score=self._unit_confidence(section.confidence_score, "api_selection_debate"),
                payload=api_selection_payload,
            )
        )

        return units

    def _unit_difficulty(self, section_difficulty: float, template_family: str) -> float:
        track = _TEMPLATE_TRACK[template_family]
        adjustment = {
            "concept": -0.4,
            "action": 0.2,
            "reasoning": 0.8,
            "uncertainty": 0.5,
        }[track]
        return round(max(1.0, min(10.0, section_difficulty + adjustment)), 2)

    def _unit_confidence(self, section_confidence: float, template_family: str) -> float:
        track = _TEMPLATE_TRACK[template_family]
        value = section_confidence
        if track == "uncertainty":
            value = min(value, 0.65)
        elif track == "reasoning":
            value = min(0.90, value + 0.03)
        elif track == "concept":
            value = min(0.92, value + 0.02)
        return round(max(0.05, min(0.95, value)), 3)

    # ------------------------------------------------------------------
    # LLM synthesis: one call per section generates Q&A for all template
    # families from the collected evidence, replacing mechanical stubs.
    # ------------------------------------------------------------------

    def _synthesize_section_qa_batch(
        self,
        section: CurriculumSection,
    ) -> dict[str, tuple[str, str]]:
        """Synthesize Q&A pairs for all template families via multi-turn tool use.

        Delegates to :class:`_MultiTurnSynthesizer`, which runs an iterative
        search / grep / head loop before producing the final training examples.
        The LLM drives its own evidence gathering: it can call up to
        ``MAX_TOOL_CALLS_PER_TURN`` tools per turn, write compressed findings
        to a persistent ``<memo>`` that carries forward, and keeps private
        reasoning in a ``<think>`` block that is discarded after each turn.

        Returns ``{}`` when no LLM is configured or synthesis fails entirely.
        """
        if self.llm_client is None:
            return {}
        synthesizer = _MultiTurnSynthesizer(
            llm_client=self.llm_client,
            retriever=self.retriever,
            cli_executor=self.cli_executor,
            repo_root=self.paths.repository,
            max_turns=6,
        )
        try:
            result = synthesizer.synthesize(section)
            if not result:
                _LOG.warning(
                    "multi-turn synthesis returned empty result for section=%s",
                    section.section_id,
                )
            return result
        except Exception as exc:
            _LOG.warning(
                "multi-turn synthesis failed for section=%s (%s)",
                section.section_id,
                exc,
            )
            return {}

    def _example_from_unit(
        self,
        section: CurriculumSection,
        unit: TopicUnit,
        synthesized_qa: dict[str, tuple[str, str]] | None = None,
    ) -> DatasetExample:
        task_category = _TEMPLATE_TO_CATEGORY[unit.template_family]
        difficulty = _bucket_difficulty(unit.difficulty_score)
        # Prefer LLM-synthesized Q&A; fall back to mechanical template assembly.
        if synthesized_qa and unit.template_family in synthesized_qa:
            question, answer = synthesized_qa[unit.template_family]
        else:
            question, answer = self._question_answer_for_unit(unit)
        question = f"{question} [section={section.section_id}; unit={unit.unit_id}]"
        contexts = self._contexts_from_evidence(unit.evidence_references, unit.supporting_files)
        if not contexts and unit.supporting_files:
            fallback = self._read_context_from_source(unit.supporting_files[0], 1, 120, "supporting_file_fallback")
            if fallback is not None:
                contexts = [fallback]

        reasoning_steps = self._reasoning_steps_for_unit(unit)
        reasoning_trace = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(reasoning_steps))
        if not reasoning_trace:
            reasoning_trace = "1. Ground claims in reopened source evidence.\n2. Mark unresolved ambiguity explicitly."

        investigation_trace = ""
        if task_category in {"agentic_task", "cli_exploration_task", "multi_hop_navigation_task", "retrieval_task"}:
            tool_sequence = unit.payload.get("tool_sequence")
            if isinstance(tool_sequence, list):
                joined = "\n".join(f"- {step}" for step in tool_sequence)
            else:
                joined = "- retriever.search\n- grep_context\n- head"
            investigation_trace = (
                f"OBJECTIVE: {unit.objective}\n"
                "KNOWN EVIDENCE: reopen source files from evidence refs\n"
                "TOOL SEQUENCE:\n"
                f"{joined}\n"
                "STOPPING CONDITION: stop after evidence saturation."
            )

        return DatasetExample(
            task_id=self._task_id_for_unit(unit.unit_id),
            task_category=task_category,
            difficulty=difficulty,
            repository_context=f"Curriculum section: {section.section_id} | objective: {section.section_objective}",
            retrieved_context=contexts,
            question=question,
            reasoning_trace=reasoning_trace,
            answer=answer,
            supporting_files=list(unit.supporting_files),
            symbols=list(unit.symbols),
            architectural_constraints=[
                "Every repository-specific claim must cite reopened source evidence.",
                "Treat investigator outputs as agenda hints, not final truth.",
                "Use retrieval for navigation; validate by reopening source/tests/config/build files.",
            ],
            validation_checks=[
                "maps_to_curriculum_section",
                "maps_to_topic_unit",
                "evidence_refs_resolve_to_source",
                "high_confidence_requires_dense_evidence",
                "difficulty_score_present",
            ],
            negative_examples=[
                (
                    unit.payload.get("tempting_but_unsupported_claim", "")
                    if unit.template_family == "contrastive_negative_example"
                    else "Do not fabricate APIs, runtime behavior, or architecture not present in evidence refs."
                )
            ],
            confidence=unit.confidence_score,
            investigation_trace=investigation_trace,
            curriculum_section_id=section.section_id,
            topic_unit_id=unit.unit_id,
            template_family=unit.template_family,
            template_fields=unit.payload,
            difficulty_score=unit.difficulty_score,
            prerequisite_task_ids=[self._task_id_for_unit(item) for item in unit.prerequisite_unit_ids],
            evidence_refs=list(unit.evidence_references),
            reasoning_steps=reasoning_steps,
            supervision_track=unit.supervision_track,
        )

    def _question_answer_for_unit(self, unit: TopicUnit) -> tuple[str, str]:
        payload = unit.payload
        family = unit.template_family

        if family == "concept_card":
            question = f"Concept Card: {payload.get('objective', unit.objective)}"
            answer = (
                f"Key concepts: {', '.join(payload.get('key_concepts', []))}. "
                f"Explanation: {payload.get('concise_explanation', '')} "
                f"Evidence: {', '.join(payload.get('supporting_evidence', []))}."
            )
            return question, answer

        if family == "api_contract_card":
            question = f"API Contract Card: {payload.get('symbol', 'unknown_symbol')}"
            answer = (
                f"Contract: {payload.get('contract_summary', '')} "
                f"Invariants: {', '.join(payload.get('invariants', []))}. "
                f"Failure behavior: {', '.join(payload.get('failure_behavior', []))}."
            )
            return question, answer

        if family == "execution_flow_trace":
            question = f"Execution Flow Trace: {payload.get('entry_point', unit.objective)}"
            answer = (
                f"Hops: {' -> '.join(payload.get('ordered_hops', []))}. "
                f"Terminal effect: {payload.get('terminal_effect', '')}. "
                f"Evidence: {', '.join(payload.get('evidence_refs', []))}."
            )
            return question, answer

        if family == "retrieval_and_tool_use_task":
            question = f"Retrieval and Tool-Use Task: {payload.get('task', unit.objective)}"
            answer = (
                f"Sequence: {' ; '.join(payload.get('tool_sequence', []))}. "
                f"Stop when: {payload.get('stopping_condition', '')}. "
                f"Grounded answer rule: {payload.get('final_grounded_answer', '')}."
            )
            return question, answer

        if family == "implementation_plan_task":
            question = f"Implementation Plan Task: {payload.get('change_goal', unit.objective)}"
            answer = (
                f"Steps: {' ; '.join(payload.get('ordered_steps', []))}. "
                f"Constraints: {' ; '.join(payload.get('constraints', []))}. "
                f"Validation: {' ; '.join(payload.get('validations', []))}."
            )
            return question, answer

        if family == "debugging_and_failure_analysis_task":
            question = f"Debugging Task: {payload.get('observed_symptom', unit.objective)}"
            answer = (
                f"Likely root cause: {payload.get('likely_root_cause', '')}. "
                f"Minimal fix: {payload.get('minimal_fix', '')}. "
                f"Validation strategy: {payload.get('validation_strategy', '')}."
            )
            return question, answer

        if family == "test_design_task":
            question = f"Test Design Task: {payload.get('target_behavior', unit.objective)}"
            answer = (
                f"Critical cases: {' ; '.join(payload.get('critical_cases', []))}. "
                f"Assertions: {' ; '.join(payload.get('assertions', []))}. "
                f"Evidence: {', '.join(payload.get('evidence_refs', []))}."
            )
            return question, answer

        if family == "contrastive_negative_example":
            question = "Contrastive Negative Example: identify why the claim is unsupported"
            answer = (
                f"Tempting claim: {payload.get('tempting_but_unsupported_claim', '')}. "
                f"Why unsupported: {payload.get('why_unsupported_or_wrong', '')}. "
                f"Correction: {payload.get('proper_cautious_answer', '')}."
            )
            return question, answer

        if family == "ambiguity_and_deferral_example":
            question = f"Ambiguity and Deferral Example: {payload.get('question', unit.objective)}"
            answer = (
                f"Safe answer: {payload.get('safe_answer', '')}. "
                f"Ambiguity source: {payload.get('ambiguity_source', '')}. "
                f"Next step: {payload.get('next_retrieval_step', '')}."
            )
            return question, answer

        if family == "design_deliberation_trace":
            question = f"Design Deliberation Trace: {payload.get('engineering_goal', unit.objective)}"
            answer = (
                f"Chosen approach: {payload.get('chosen_approach', '')}. "
                f"Tradeoffs: {' ; '.join(payload.get('tradeoffs', []))}. "
                f"Validation plan: {' ; '.join(payload.get('validation_plan', []))}."
            )
            return question, answer

        if family == "best_practice_transfer_card":
            question = f"Best-Practice Transfer Card: {payload.get('practice', unit.objective)}"
            answer = (
                f"Where observed: {', '.join(payload.get('where_observed', []))}. "
                f"Apply when: {payload.get('when_to_apply', '')}. "
                f"Do not apply when: {payload.get('when_not_to_apply', '')}."
            )
            return question, answer

        if family == "api_selection_debate":
            question = f"API Selection Debate: {payload.get('task', unit.objective)}"
            answer = (
                f"Candidates: {', '.join(payload.get('candidate_apis', []))}. "
                f"Selected API: {payload.get('selected_api', '')}. "
                f"Rationale: {payload.get('rationale', '')}."
            )
            return question, answer

        return unit.objective, "Insufficient template mapping; defer until fields are clarified."

    def _reasoning_steps_for_unit(self, unit: TopicUnit) -> list[str]:
        payload = unit.payload
        if unit.template_family == "design_deliberation_trace":
            return [
                f"Clarify engineering goal: {payload.get('engineering_goal', unit.objective)}",
                "Enumerate candidate approaches with compatibility constraints.",
                "Compare tradeoffs using reopened code precedents.",
                f"Select approach and validate via: {' ; '.join(payload.get('validation_plan', []))}.",
            ]
        if unit.template_family == "api_selection_debate":
            return [
                f"List candidate APIs: {', '.join(payload.get('candidate_apis', []))}",
                "Check contract compatibility and call-site precedent.",
                f"Select API '{payload.get('selected_api', '')}' with explicit risks.",
                "Document rejected options and validation checks.",
            ]
        if unit.supervision_track == "reasoning":
            return [
                "Start from concrete evidence refs.",
                "Evaluate alternatives against repository conventions.",
                "Choose the lowest-risk approach and define validation steps.",
            ]
        return []

    def _contexts_from_evidence(
        self,
        evidence_refs: tuple[str, ...],
        supporting_files: tuple[str, ...],
    ) -> list[dict[str, object]]:
        contexts: list[dict[str, object]] = []
        seen_paths: set[str] = set()

        for ref in evidence_refs:
            parsed = _parse_source_ref(ref)
            if parsed is None:
                continue
            path, start_line, end_line = parsed
            context = self._read_context_from_source(path, start_line, end_line, "evidence_ref")
            if context is None:
                continue
            contexts.append(context)
            seen_paths.add(path)
            if len(contexts) >= 12:
                return contexts

        for path in supporting_files:
            if len(contexts) >= 12:
                break
            if path in seen_paths:
                continue
            context = self._read_context_from_source(path, 1, 120, "supporting_file")
            if context is None:
                continue
            contexts.append(context)
        return contexts

    def _build_coverage_map(
        self,
        sections: list[CurriculumSection],
        units: list[TopicUnit],
        dataset_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        section_examples = Counter(str(row.get("curriculum_section_id", "")) for row in dataset_rows)
        template_counts = Counter(str(row.get("template_family", "")) for row in dataset_rows)
        track_counts = Counter(str(row.get("supervision_track", "")) for row in dataset_rows)
        difficulty_counts = Counter(str(row.get("difficulty", "")) for row in dataset_rows)

        topic_coverage: Counter[str] = Counter()
        for section in sections:
            for topic in section.investigator_topics:
                topic_coverage[topic] += section_examples.get(section.section_id, 0)

        section_map: dict[str, Any] = {}
        for section in sections:
            section_map[section.section_id] = {
                "examples": section_examples.get(section.section_id, 0),
                "units": sum(1 for unit in units if unit.section_id == section.section_id),
                "evidence_refs": len(section.evidence_references),
                "difficulty_score": section.difficulty_score,
                "confidence_score": section.confidence_score,
                "investigator_topics": list(section.investigator_topics),
            }

        return {
            "schema": "distillme.coverage_map.v1",
            "section_coverage": section_map,
            "investigator_topic_coverage": dict(sorted(topic_coverage.items())),
            "template_family_coverage": dict(sorted(template_counts.items())),
            "supervision_track_coverage": dict(sorted(track_counts.items())),
            "difficulty_distribution": dict(sorted(difficulty_counts.items())),
        }

    def _build_teacher_report(
        self,
        agenda: list[InvestigatorFindingRecord],
        sections: list[CurriculumSection],
        units: list[TopicUnit],
        dataset_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        high_conf_examples = [row for row in dataset_rows if float(row.get("confidence", 0.0)) >= HIGH_CONFIDENCE_THRESHOLD]
        avg_iterations = round(
            sum(section.compilation_iterations for section in sections) / max(len(sections), 1),
            2,
        )
        return {
            "schema": "distillme.teacher_report.v1",
            "investigator_records": len(agenda),
            "sections_compiled": len(sections),
            "topic_units_generated": len(units),
            "examples_generated": len(dataset_rows),
            "average_section_iterations": avg_iterations,
            "high_confidence_examples": len(high_conf_examples),
            "notes": [
                "Investigator records are treated as agenda-setting hints.",
                "Retrieval hits influence curriculum only after source reopen validation.",
                "Section compilation uses adaptive top_k (8 -> 12/16 -> 24) with evidence saturation stop.",
            ],
        }

    def _section_from_json(self, row: dict[str, Any]) -> CurriculumSection:
        return CurriculumSection(
            section_id=str(row.get("section_id", "")),
            section_objective=str(row.get("section_objective", "")),
            prerequisite_sections=tuple(row.get("prerequisite_sections", [])),
            core_files=tuple(row.get("core_files", [])),
            core_symbols=tuple(row.get("core_symbols", [])),
            api_contracts=tuple(row.get("api_contracts", [])),
            recurring_codebase_conventions=tuple(row.get("recurring_codebase_conventions", [])),
            best_practice_patterns=tuple(row.get("best_practice_patterns", [])),
            invariants=tuple(row.get("invariants", [])),
            execution_flows=tuple(row.get("execution_flows", [])),
            failure_modes=tuple(row.get("failure_modes", [])),
            evidence_references=tuple(row.get("evidence_references", [])),
            confidence_score=float(row.get("confidence_score", 0.0)),
            ambiguity_open_questions=tuple(row.get("ambiguity_open_questions", [])),
            recommended_task_families=tuple(row.get("recommended_task_families", [])),
            difficulty_score=float(row.get("difficulty_score", 0.0)),
            investigator_topics=tuple(row.get("investigator_topics", [])),
            compilation_iterations=int(row.get("compilation_iterations", 0)),
        )

    def _topic_unit_from_json(self, row: dict[str, Any]) -> TopicUnit:
        return TopicUnit(
            unit_id=str(row.get("unit_id", "")),
            section_id=str(row.get("section_id", "")),
            objective=str(row.get("objective", "")),
            template_family=str(row.get("template_family", "concept_card")),
            supervision_track=str(row.get("supervision_track", "concept")),
            prerequisite_unit_ids=tuple(row.get("prerequisite_unit_ids", [])),
            evidence_references=tuple(row.get("evidence_references", [])),
            supporting_files=tuple(row.get("supporting_files", [])),
            symbols=tuple(row.get("symbols", [])),
            difficulty_score=float(row.get("difficulty_score", 0.0)),
            confidence_score=float(row.get("confidence_score", 0.0)),
            payload=dict(row.get("payload", {})),
        )

    def _task_id_for_unit(self, unit_id: str) -> str:
        return f"unit::{unit_id.replace('::', '--')}"

    def _unit_id(self, section_id: str, template_family: str, index: int) -> str:
        return f"{section_id}::{template_family}::{index:02d}"

    def _concept_unit_id(self, section_id: str) -> str:
        return self._unit_id(section_id, "concept_card", 1)

    def _extract_scalar(self, text: str, prefix: str, default: str = "") -> str:
        for line in text.splitlines():
            if line.startswith(prefix):
                return line.removeprefix(prefix).strip()
        return default

    def _extract_bullets(self, text: str, heading: str) -> list[str]:
        lines = text.splitlines()
        items: list[str] = []
        in_block = False
        for line in lines:
            if line.strip() == heading:
                in_block = True
                continue
            if not in_block:
                continue
            if line.startswith("  - ") or line.startswith("    - "):
                item = line.strip()[2:].strip().strip("`")
                if item:
                    items.append(item)
                continue
            if line.startswith("- "):
                break
            if line.startswith("## ") or line.startswith("### "):
                break
            if not line.strip():
                continue
            if line.strip().startswith("**") and line.strip().endswith("**"):
                break
        return items

    def _extract_scalar_from_lines(self, lines: list[str], prefix: str) -> str:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped.removeprefix(prefix).strip()
        return ""

    def _extract_bullets_from_lines(self, lines: list[str], heading: str) -> list[str]:
        items: list[str] = []
        in_block = False
        for line in lines:
            stripped = line.strip()
            if stripped == heading:
                in_block = True
                continue
            if not in_block:
                continue
            if line.startswith("    - ") or line.startswith("  - "):
                item = stripped[2:].strip().strip("`")
                if item:
                    items.append(item)
                continue
            if stripped.startswith("**"):
                break
            if not stripped:
                continue
            if stripped.startswith("#### "):
                break
        return items

    def _extract_section(self, text: str, start_heading: str, end_headings: tuple[str, ...]) -> str:
        lines = text.splitlines()
        start = None
        for index, line in enumerate(lines):
            if line.strip() == start_heading:
                start = index + 1
                break
        if start is None:
            return ""
        collected: list[str] = []
        for line in lines[start:]:
            stripped = line.strip()
            if any(stripped == heading for heading in end_headings):
                break
            if stripped.startswith("## ") and stripped != start_heading:
                break
            collected.append(line)
        return "\n".join(collected).strip()

    def _safe_repo_file(self, rel_path: str) -> Path | None:
        candidate = Path(rel_path)
        if not candidate.is_absolute():
            candidate = (self.paths.repository / rel_path).resolve()
        else:
            candidate = candidate.resolve()
        repo_root = self.paths.repository.resolve()
        if candidate == repo_root or repo_root in candidate.parents:
            return candidate
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_partial_map(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        key = str(row.get(id_field, "")).strip()
        if key and key not in mapping:
            mapping[key] = row
    return mapping


def _normalize_rel_path(path: str) -> str:
    candidate = path.strip().strip("`").replace("\\", "/")
    if candidate.startswith("./"):
        candidate = candidate[2:]
    return candidate


def _parse_source_ref(text: str) -> tuple[str, int, int] | None:
    cleaned = text.strip().strip("`")
    match = _SOURCE_REF_RE.match(cleaned)
    if match is None:
        return None
    path = _normalize_rel_path(match.group("path"))
    if not path:
        return None
    start = int(match.group("start"))
    end_raw = match.group("end")
    end = int(end_raw) if end_raw else start
    return path, start, max(start, end)


def _extract_paths_from_text(text: str) -> list[str]:
    paths: list[str] = []
    for match in _PATH_IN_TEXT_RE.findall(text):
        path = _normalize_rel_path(match)
        if path:
            paths.append(path)
    return paths


def _extract_symbols(text: str) -> list[str]:
    symbols = {symbol for _, symbol in _SYMBOL_DEF_RE.findall(text)}
    return sorted(symbols)


def _extract_symbol_from_contract(contract: str) -> str:
    candidate = contract.split(":", 1)[-1].strip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)", candidate)
    if match:
        return match.group(1)
    return "unknown_symbol"


def _bucket_difficulty(score: float) -> str:
    if score < 3.0:
        return "easy"
    if score < 5.0:
        return "medium"
    if score < 7.0:
        return "hard"
    return "expert"


def _limit_sorted(values: set[str], limit: int) -> list[str]:
    return sorted(values)[:limit]


def _add_capped(values: set[str], item: str, limit: int) -> bool:
    if not item or item in values:
        return False
    if len(values) >= limit:
        return False
    values.add(item)
    return True
