"""Grounding and dataset quality validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from distillme.schemas import PipelinePaths


class ValidationPipeline:
    """Performs deterministic quality checks before training."""

    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths

    def run(self) -> dict[str, int | float]:
        examples = _load_jsonl(self.paths.dataset_dir / "instruction_dataset.jsonl")
        artifacts = _load_jsonl(self.paths.index_dir / "artifacts.jsonl")
        known_files = {row["path"] for row in artifacts}
        failures: list[dict[str, str]] = []
        seen_questions: set[str] = set()
        for example in examples:
            task_id = str(example.get("task_id", "unknown"))
            files = set(example.get("supporting_files", []))
            missing = files - known_files
            if missing:
                failures.append({"task_id": task_id, "reason": f"unknown supporting files: {sorted(missing)}"})
            if example.get("question") in seen_questions:
                failures.append({"task_id": task_id, "reason": "duplicate question"})
            seen_questions.add(str(example.get("question")))
            if files and not example.get("retrieved_context"):
                failures.append({"task_id": task_id, "reason": "missing retrieved context"})
            if "Uncertain" not in str(example.get("answer", "")) and "uncertain" not in str(example.get("answer", "")):
                failures.append({"task_id": task_id, "reason": "answer lacks uncertainty calibration"})
        self.paths.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.paths.dataset_dir / "validation_report.json").write_text(
            json.dumps(
                {
                    "examples": len(examples),
                    "failures": failures,
                    "passed": not failures,
                    "hallucination_proxy_rate": len(failures) / max(len(examples), 1),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"examples": len(examples), "failures": len(failures)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
