from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillme.config import PipelineConfig
from distillme.orchestration import DistillationPipeline
from distillme.retrieval import HybridRetriever


def _write_sample_repo(root: Path) -> None:
    (root / "src/main/java/com/example").mkdir(parents=True)
    (root / "src/test/java/com/example").mkdir(parents=True)
    (root / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
    (root / "src/main/java/com/example/Greeter.java").write_text(
        """
package com.example;

public class Greeter {
    public String greet(String name) {
        if (name == null) {
            throw new IllegalArgumentException("name");
        }
        return "Hello " + name;
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "src/test/java/com/example/GreeterTest.java").write_text(
        """
package com.example;

class GreeterTest {
    void greets() {
        new Greeter().greet("repo");
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_config_rejects_correlated_model_families(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = PipelineConfig.default(repo, tmp_path / "work")
    bad = PipelineConfig(
        repository_path=config.repository_path,
        workdir=config.workdir,
        investigator=config.investigator,
        teacher=config.teacher,
        student=config.teacher,
    )
    with pytest.raises(ValueError, match="must be distinct"):
        bad.validate()


def test_pipeline_generates_grounded_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_sample_repo(repo)
    config = PipelineConfig.default(repo, tmp_path / "work")

    results = DistillationPipeline(config).run(resume=False)

    assert all(result.status == "succeeded" for result in results.values())
    assert (config.workdir / "index/manifest.json").exists()
    assert (config.workdir / "investigator/architecture_overview.md").exists()
    dataset_path = config.workdir / "dataset/instruction_dataset.jsonl"
    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    assert records
    assert all(record["supporting_files"] for record in records)
    report = json.loads((config.workdir / "dataset/validation_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_hybrid_retriever_finds_symbols_after_ingest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_sample_repo(repo)
    config = PipelineConfig.default(repo, tmp_path / "work")
    DistillationPipeline(config).run(resume=False)

    hits = HybridRetriever(config.workdir / "index").search("Greeter greet", top_k=2)

    assert hits
    assert hits[0].chunk.path.endswith("Greeter.java")
