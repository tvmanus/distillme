"""Training and evaluation planning artifacts for the 7B student model."""

from __future__ import annotations

import json
from distillme.config import PipelineConfig
from distillme.schemas import PipelinePaths


class StudentTrainingPlanner:
    """Writes reproducible QLoRA/SFT/DPO training plans without assuming a GPU stack."""

    def __init__(self, config: PipelineConfig, paths: PipelinePaths) -> None:
        self.config = config
        self.paths = paths

    def run(self) -> dict[str, int]:
        self.paths.training_dir.mkdir(parents=True, exist_ok=True)
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
            "safety": [
                "cite_retrieved_evidence",
                "defer_when_uncertain",
                "preserve_coding_conventions",
                "reject_unsupported_repository_claims",
            ],
        }
        (self.paths.training_dir / "training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return {"plans": 1}


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
