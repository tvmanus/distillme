"""Training and evaluation planning artifacts for the 7B student model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from distillme.config import PipelineConfig
from distillme.schemas import PipelinePaths


class StudentTrainingPlanner:
    """Writes reproducible QLoRA/SFT/DPO training plans without assuming a GPU stack."""

    def __init__(self, config: PipelineConfig, paths: PipelinePaths) -> None:
        self.config = config
        self.paths = paths

    def run(self) -> dict[str, int]:
        self.paths.training_dir.mkdir(parents=True, exist_ok=True)
        examples = _load_jsonl(self.paths.dataset_dir / "instruction_dataset.jsonl")
        curriculum = _load_json(self.paths.dataset_dir / "curriculum.json")
        topic_units = _load_json(self.paths.dataset_dir / "topic_units.json")
        coverage_map = _load_json(self.paths.dataset_dir / "coverage_map.json")

        template_counts = Counter(str(row.get("template_family", "")) for row in examples)
        track_counts = Counter(str(row.get("supervision_track", "")) for row in examples)

        plan = {
            "student_model": self.config.student.model,
            "student_family": self.config.student.family,
            "strategy": ["supervised_fine_tuning", "qlora", "curriculum_learning", "retrieval_aware_fine_tuning"],
            "splits": {"train": 0.80, "validation": 0.10, "test": 0.10},
            "objectives": [
                "repository_qa",
                "architectural_reasoning",
                "code_completion",
                "implementation_planning",
                "debugging_guidance",
                "retrieval_grounded_responses",
                "tool_aware_reasoning",
            ],
            "curriculum_artifacts": {
                "curriculum_sections": len(curriculum.get("sections", [])),
                "topic_units": len(topic_units.get("units", [])),
                "coverage_map_present": bool(coverage_map),
                "instruction_examples": len(examples),
            },
            "tracks": {
                "concept_teaching": int(track_counts.get("concept", 0)),
                "action_planning": int(track_counts.get("action", 0)),
                "reasoning_supervision": int(track_counts.get("reasoning", 0)),
                "uncertainty_handling": int(track_counts.get("uncertainty", 0)),
            },
            "template_distribution": dict(sorted(template_counts.items())),
            "curriculum_sampling_policy": {
                "progression": "prerequisite_first_then_difficulty",
                "difficulty_order": ["easy", "medium", "hard", "expert"],
                "high_confidence_priority": True,
                "reasoning_trace_budget": "bounded_stepwise",
            },
            "reasoning_supervision": {
                "enabled": True,
                "format": "concise_stepwise_rationale",
                "required_for": ["design_deliberation_trace", "api_selection_debate", "implementation_plan_task"],
                "rejected_format": "unbounded_monologue",
            },
            "safety": [
                "cite_retrieved_evidence",
                "defer_when_uncertain",
                "preserve_coding_conventions",
                "reject_unsupported_repository_claims",
            ],
        }
        (self.paths.training_dir / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return {
            "plans": 1,
            "examples": len(examples),
            "sections": len(curriculum.get("sections", [])),
            "units": len(topic_units.get("units", [])),
        }


class EvaluationPlanner:
    """Writes benchmark definitions for specialist advisor evaluation."""

    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths

    def run(self) -> dict[str, int]:
        self.paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
        benchmarks = {
            "benchmarks": [
                "repository_qa_accuracy",
                "implementation_correctness",
                "compile_success_rate",
                "architectural_consistency",
                "hallucination_rate",
                "retrieval_precision",
                "bug_fix_success_rate",
                "test_generation_quality",
                "code_style_consistency",
                "latency",
                "token_efficiency",
                "long_context_retention",
            ],
            "required_reports": ["coverage", "failure_cases", "cross_model_disagreements", "regressions"],
        }
        (self.paths.evaluation_dir / "benchmarks.json").write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")
        return {"benchmarks": len(benchmarks["benchmarks"])}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
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
