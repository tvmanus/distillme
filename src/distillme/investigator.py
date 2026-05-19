"""Investigator stage that emits evidence-backed architecture documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from distillme.retrieval import RetrievalHit
from distillme.retrieval import HybridRetriever
from distillme.schemas import Finding, PipelinePaths

if TYPE_CHECKING:
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


class InvestigatorAgent:
    """Produces the required finding documents from indexed evidence."""

    def __init__(
        self,
        paths: PipelinePaths,
        retriever: HybridRetriever,
        llm_client: "LLMClient | None" = None,
    ) -> None:
        self.paths = paths
        self.retriever = retriever
        self.llm_client = llm_client

    def run(self) -> dict[str, int]:
        self.paths.investigator_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for document in MANDATORY_DOCUMENTS:
            query = _DOCUMENT_QUERIES[document]
            hits = self.retriever.search(query, top_k=12)
            finding = self._finding_for(document, query, hits)
            model_analysis = self._model_analysis(document, hits)
            target = self.paths.investigator_dir / document
            target.write_text(
                _render_document(document, query, finding, model_analysis),
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


def _render_document(document: str, query: str, finding: Finding, model_analysis: str) -> str:
    return (
        f"# {document.removesuffix('.md').replace('_', ' ').title()}\n\n"
        "Generated by the Investigator Agent. All claims are intentionally evidence-scoped.\n\n"
        f"Investigation seed: `{query}`\n\n"
        f"{finding.to_markdown()}\n"
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
