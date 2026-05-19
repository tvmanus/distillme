"""Resumable DAG orchestration for the three-stage distillation pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from distillme.config import PipelineConfig
from distillme.inference import make_client
from distillme.ingestion import RepositoryIngestor
from distillme.investigator import InvestigatorAgent
from distillme.observability import TraceLogger
from distillme.retrieval import HybridRetriever
from distillme.schemas import PipelinePaths, StageName, StageResult
from distillme.teacher import TeacherAgent
from distillme.training import EvaluationPlanner, StudentTrainingPlanner
from distillme.validation import ValidationPipeline

STAGES: tuple[StageName, ...] = ("ingest", "investigate", "teach", "validate", "train", "evaluate")


class DistillationPipeline:
    """Coordinates ingestion, investigation, teaching, validation, training, and evaluation."""

    def __init__(self, config: PipelineConfig) -> None:
        config.validate()
        self.config = config
        self.paths = PipelinePaths.from_root(config.repository_path, config.workdir)
        self.paths.create()
        self.trace = TraceLogger(self.paths.logs_dir)
        self.state_path = self.paths.workdir / "state.json"

    def run(self, resume: bool = True) -> dict[str, StageResult]:
        state = self._load_state() if resume else {}
        results: dict[str, StageResult] = {}
        for stage in STAGES:
            if resume and state.get(stage, {}).get("status") == "succeeded":
                results[stage] = StageResult(stage=stage, status="succeeded", metrics=state[stage].get("metrics", {}))
                continue
            result = self._run_stage(stage)
            results[stage] = result
            state[stage] = _result_to_json(result)
            self._save_state(state)
            if result.status != "succeeded":
                break
        return results

    def _run_stage(self, stage: StageName) -> StageResult:
        started = time.time()
        self.trace.event("stage_started", stage=stage)
        try:
            runner = self._runner_for(stage)
            metrics = runner()
            elapsed = time.time() - started
            metrics = {**metrics, "elapsed_seconds": round(elapsed, 4)}
            self.trace.event("stage_succeeded", stage=stage, metrics=metrics)
            return StageResult(stage=stage, status="succeeded", metrics=metrics)
        except Exception as exc:  # pragma: no cover - exercised by integration failure paths
            self.trace.event("stage_failed", stage=stage, error=str(exc))
            return StageResult(stage=stage, status="failed", error=str(exc))

    def _runner_for(self, stage: StageName) -> Callable[[], dict[str, int | float]]:
        if stage == "ingest":
            return RepositoryIngestor(self.config, self.paths).run
        if stage == "investigate":
            return InvestigatorAgent(self.paths, self._retriever(), make_client(self.config.investigator)).run
        if stage == "teach":
            return TeacherAgent(self.paths, self._retriever(), make_client(self.config.teacher)).run
        if stage == "validate":
            return ValidationPipeline(self.paths).run
        if stage == "train":
            return StudentTrainingPlanner(self.config, self.paths).run
        if stage == "evaluate":
            return EvaluationPlanner(self.paths).run
        raise ValueError(f"unknown stage: {stage}")

    def _retriever(self) -> HybridRetriever:
        retrieval = self.config.retrieval
        return HybridRetriever(
            self.paths.index_dir,
            dense_weight=retrieval.dense_weight,
            sparse_weight=retrieval.sparse_weight,
            symbol_weight=retrieval.symbol_weight,
        )

    def _load_state(self) -> dict[str, dict[str, object]]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict[str, dict[str, object]]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _result_to_json(result: StageResult) -> dict[str, object]:
    return asdict(result)
