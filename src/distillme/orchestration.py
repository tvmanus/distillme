"""Resumable DAG orchestration for the three-stage distillation pipeline."""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from distillme.cli_tools import CliExecutor
from distillme.config import PipelineConfig
from distillme.inference import make_exclusive_client
from distillme.ingestion import RepositoryIngestor
from distillme.investigator import InvestigatorAgent
from distillme.observability import TraceLogger
from distillme.retrieval import ChromaRetriever, HybridRetriever, make_retriever
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
        self._cli = CliExecutor(config.repository_path)

    def run(self, resume: bool = True, from_stage: StageName | None = None) -> dict[str, StageResult]:
        state = self._load_state() if resume else {}

        # --from-stage: drop state for that stage and everything after it so those
        # stages are unconditionally re-run.  Earlier stages are left intact so their
        # outputs are preserved and skipped as usual.
        if from_stage is not None:
            if from_stage not in STAGES:
                raise ValueError(f"unknown stage: {from_stage!r}; valid stages: {STAGES}")
            restart_idx = STAGES.index(from_stage)
            for s in STAGES[restart_idx:]:
                state.pop(s, None)
            self._save_state(state)

        # Intra-stage resume (skip already-written documents, reload partial files) only
        # makes sense for a naturally-interrupted run.  --from-stage and --no-resume both
        # start each affected stage from a clean slate.
        intra_resume = resume and from_stage is None

        results: dict[str, StageResult] = {}
        for stage in STAGES:
            if resume and state.get(stage, {}).get("status") == "succeeded":
                results[stage] = StageResult(stage=stage, status="succeeded", metrics=state[stage].get("metrics", {}))
                continue

            # Persist a start breadcrumb so a crash leaves a timestamp in state.json.
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            state.setdefault(stage, {})["started_at"] = now
            self._save_state(state)

            result = self._run_stage(stage, resume=intra_resume)
            results[stage] = result
            state[stage] = {
                **_result_to_json(result),
                "started_at": state.get(stage, {}).get("started_at"),
                "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            }
            self._save_state(state)
            if result.status != "succeeded":
                break
        return results

    def _run_stage(self, stage: StageName, resume: bool = True) -> StageResult:
        started = time.time()
        self.trace.event("stage_started", stage=stage)
        try:
            runner = self._runner_for(stage, resume=resume)
            metrics = runner()
            elapsed = time.time() - started
            metrics = {**metrics, "elapsed_seconds": round(elapsed, 4)}
            self.trace.event("stage_succeeded", stage=stage, metrics=metrics)
            return StageResult(stage=stage, status="succeeded", metrics=metrics)
        except Exception as exc:  # pragma: no cover - exercised by integration failure paths
            self.trace.event("stage_failed", stage=stage, error=str(exc))
            return StageResult(stage=stage, status="failed", error=str(exc))

    def _runner_for(self, stage: StageName, resume: bool = True) -> Callable[[], dict[str, int | float]]:
        if stage == "ingest":
            return RepositoryIngestor(self.config, self.paths).run
        if stage == "investigate":
            agent = InvestigatorAgent(self.paths, self._retriever(), make_exclusive_client(self.config.investigator), self._cli)
            return lambda: agent.run(resume=resume)
        if stage == "teach":
            agent = TeacherAgent(self.paths, self._retriever(), make_exclusive_client(self.config.teacher), self._cli)
            return lambda: agent.run(resume=resume)
        if stage == "validate":
            return ValidationPipeline(self.paths).run
        if stage == "train":
            return StudentTrainingPlanner(self.config, self.paths).run
        if stage == "evaluate":
            return EvaluationPlanner(self.paths).run
        raise ValueError(f"unknown stage: {stage}")

    def _retriever(self) -> "HybridRetriever | ChromaRetriever":
        return make_retriever(self.config.retrieval, self.paths.index_dir)

    def _load_state(self) -> dict[str, dict[str, object]]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict[str, dict[str, object]]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _result_to_json(result: StageResult) -> dict[str, object]:
    return asdict(result)
