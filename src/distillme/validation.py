"""Grounding and dataset quality validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from distillme.schemas import PipelinePaths

UNCERTAINTY_LANGUAGE = ("uncertain", "not sure", "possibly", "maybe", "unclear", "insufficient", "unknown", "unsupported", "cannot confirm", "contradict")
HIGH_CONFIDENCE_THRESHOLD = 0.78


class ValidationPipeline:
    """Performs deterministic quality checks before training."""

    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths

    def run(self) -> dict[str, int | float]:
        examples = _load_jsonl(self.paths.dataset_dir / "instruction_dataset.jsonl")
        artifacts = _load_jsonl(self.paths.index_dir / "artifacts.jsonl")
        curriculum = _load_json(self.paths.dataset_dir / "curriculum.json")
        topic_units_payload = _load_json(self.paths.dataset_dir / "topic_units.json")
        coverage_map = _load_json(self.paths.dataset_dir / "coverage_map.json")

        sections = {str(row.get("section_id", "")): row for row in curriculum.get("sections", [])}
        units = {str(row.get("unit_id", "")): row for row in topic_units_payload.get("units", [])}

        known_files = {row["path"] for row in artifacts}
        failures: list[dict[str, str]] = []
        seen_questions: set[str] = set()
        score_groups: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
        topic_coverage: Counter[str] = Counter()

        for example in examples:
            task_id = str(example.get("task_id", "unknown"))
            files = set(example.get("supporting_files", []))
            missing = files - known_files
            if missing:
                failures.append({"task_id": task_id, "reason": f"unknown supporting files: {sorted(missing)}"})
            if example.get("question") in seen_questions:
                failures.append({"task_id": task_id, "reason": "duplicate question"})
            seen_questions.add(str(example.get("question")))
            retrieved_context = example.get("retrieved_context") or []
            if files and not retrieved_context:
                failures.append({"task_id": task_id, "reason": "missing retrieved context"})

            section_id = str(example.get("curriculum_section_id", ""))
            unit_id = str(example.get("topic_unit_id", ""))
            if not section_id:
                failures.append({"task_id": task_id, "reason": "missing curriculum_section_id"})
            elif section_id not in sections:
                failures.append({"task_id": task_id, "reason": f"unknown curriculum_section_id: {section_id}"})

            if not unit_id:
                failures.append({"task_id": task_id, "reason": "missing topic_unit_id"})
            elif unit_id not in units:
                failures.append({"task_id": task_id, "reason": f"unknown topic_unit_id: {unit_id}"})
            elif section_id and unit_id in units:
                expected_section = str(units[unit_id].get("section_id", ""))
                if section_id != expected_section:
                    failures.append(
                        {
                            "task_id": task_id,
                            "reason": (
                                f"topic_unit_id points to section '{expected_section}' but example references '{section_id}'"
                            ),
                        }
                    )

            if section_id in sections:
                for topic in sections[section_id].get("investigator_topics", []):
                    topic_coverage[str(topic)] += 1

            confidence = float(example.get("confidence", 0.0))
            answer = str(example.get("answer", "")).lower()
            answer_expresses_uncertainty = any(indicator in answer for indicator in UNCERTAINTY_LANGUAGE)
            template_family = str(example.get("template_family", ""))
            uncertainty_required_families = {"contrastive_negative_example", "ambiguity_and_deferral_example"}
            if confidence < 0.35 and template_family in uncertainty_required_families and not answer_expresses_uncertainty:
                failures.append({"task_id": task_id, "reason": "answer lacks uncertainty calibration"})

            evidence_refs = example.get("evidence_refs") or []
            if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                if len(evidence_refs) < 3:
                    failures.append(
                        {"task_id": task_id, "reason": "high-confidence example lacks dense evidence refs (>=3)"}
                    )
                if len(retrieved_context) < 2:
                    failures.append(
                        {"task_id": task_id, "reason": "high-confidence example has insufficient retrieved context"}
                    )

            template_family = str(example.get("template_family", ""))
            if template_family == "contrastive_negative_example":
                if not evidence_refs:
                    failures.append({"task_id": task_id, "reason": "contrastive example has no evidence refs"})
                if not any(token in answer for token in ("unsupported", "contradict", "cannot confirm", "insufficient")):
                    failures.append(
                        {
                            "task_id": task_id,
                            "reason": "contrastive example answer must explicitly negate unsupported claim",
                        }
                    )

            difficulty_score = float(example.get("difficulty_score", 0.0))
            if section_id and template_family:
                score_groups[(section_id, template_family)].append(difficulty_score)
            if example.get("difficulty") != _bucket_difficulty(difficulty_score):
                failures.append(
                    {
                        "task_id": task_id,
                        "reason": (
                            "difficulty label is inconsistent with difficulty_score bucket"
                        ),
                    }
                )

        for (section_id, template_family), scores in score_groups.items():
            if len(scores) < 2:
                continue
            spread = max(scores) - min(scores)
            if spread > 3.5:
                failures.append(
                    {
                        "task_id": f"{section_id}:{template_family}",
                        "reason": f"difficulty spread too wide within section/template family ({spread:.2f})",
                    }
                )

        if topic_coverage:
            counts = [count for count in topic_coverage.values() if count > 0]
            if counts and max(counts) > 4 * max(1, min(counts)):
                failures.append(
                    {
                        "task_id": "coverage_balance",
                        "reason": "coverage is heavily imbalanced across investigator topics",
                    }
                )

        if sections and coverage_map:
            covered_sections = set(coverage_map.get("section_coverage", {}).keys())
            missing_coverage_sections = set(sections) - covered_sections
            if missing_coverage_sections:
                failures.append(
                    {
                        "task_id": "coverage_map",
                        "reason": f"coverage_map missing sections: {sorted(missing_coverage_sections)}",
                    }
                )

        self.paths.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.paths.dataset_dir / "validation_report.json").write_text(
            json.dumps(
                {
                    "examples": len(examples),
                    "sections": len(sections),
                    "units": len(units),
                    "failures": failures,
                    "passed": not failures,
                    "hallucination_proxy_rate": len(failures) / max(len(examples), 1),
                    "topic_coverage": dict(sorted(topic_coverage.items())),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "examples": len(examples),
            "sections": len(sections),
            "units": len(units),
            "failure_count": len(failures),
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _bucket_difficulty(score: float) -> str:
    if score < 3.0:
        return "easy"
    if score < 5.0:
        return "medium"
    if score < 7.0:
        return "hard"
    return "expert"
