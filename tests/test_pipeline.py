from __future__ import annotations

import json
import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distillme.config import PipelineConfig
from distillme.inference import HttpLLMClient, LLMClient, StubLLMClient, make_client
from distillme.orchestration import STAGES, DistillationPipeline
from distillme.retrieval import HybridRetriever
from distillme.schemas import ModelSpec
from distillme.teacher import DIFFICULTIES, TASK_CATEGORIES


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


class InferenceTests(unittest.TestCase):
    def test_stub_client_returns_deterministic_text(self) -> None:
        spec = ModelSpec(role="investigator", family="gemini", model="gemini-stub", endpoint="local")
        client = StubLLMClient(spec)
        result = client.generate("sys", "user")
        self.assertIsInstance(result, str)
        self.assertIn("stub:gemini/gemini-stub", result)
        self.assertEqual(result, client.generate("sys", "user"))

    def test_make_client_returns_stub_for_non_http_endpoint(self) -> None:
        spec = ModelSpec(role="investigator", family="gemini", model="m", endpoint="local")
        self.assertIsInstance(make_client(spec), StubLLMClient)

    def test_make_client_returns_http_for_http_endpoint(self) -> None:
        spec = ModelSpec(role="investigator", family="gemini", model="m", endpoint="http://localhost:8080")
        self.assertIsInstance(make_client(spec), HttpLLMClient)

    def test_llm_client_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            LLMClient()  # type: ignore[abstract]


class PipelineTests(unittest.TestCase):
    def test_config_rejects_correlated_model_families(self) -> None:
        with self.subTest("same teacher and student family is rejected"):
            with tempfile.TemporaryDirectory() as directory:
                tmp_path = Path(directory)
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
                with self.assertRaisesRegex(ValueError, "must be distinct"):
                    bad.validate()

    def test_pipeline_generates_grounded_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            repo = tmp_path / "repo"
            repo.mkdir()
            _write_sample_repo(repo)
            config = PipelineConfig.default(repo, tmp_path / "work")

            results = DistillationPipeline(config).run(resume=False)

            self.assertEqual(set(results), set(STAGES))
            self.assertTrue(all(result.status == "succeeded" for result in results.values()))
            self.assertTrue((config.workdir / "index/manifest.json").exists())
            self.assertTrue((config.workdir / "investigator/architecture_overview.md").exists())
            dataset_path = config.workdir / "dataset/instruction_dataset.jsonl"
            records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(records)
            self.assertTrue(all(record["supporting_files"] for record in records))
            report = json.loads((config.workdir / "dataset/validation_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])

    def test_dataset_covers_all_categories_and_difficulties(self) -> None:
        """Each (category, difficulty) pair must appear exactly once."""
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            repo = tmp_path / "repo"
            repo.mkdir()
            _write_sample_repo(repo)
            config = PipelineConfig.default(repo, tmp_path / "work")
            DistillationPipeline(config).run(resume=False)

            dataset_path = config.workdir / "dataset/instruction_dataset.jsonl"
            records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
            expected = len(TASK_CATEGORIES) * len(DIFFICULTIES)
            self.assertEqual(len(records), expected, f"expected {expected} examples, got {len(records)}")
            seen_categories = {r["task_category"] for r in records}
            seen_difficulties = {r["difficulty"] for r in records}
            self.assertEqual(seen_categories, set(TASK_CATEGORIES))
            self.assertEqual(seen_difficulties, set(DIFFICULTIES))

    def test_investigator_documents_include_model_analysis_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            repo = tmp_path / "repo"
            repo.mkdir()
            _write_sample_repo(repo)
            config = PipelineConfig.default(repo, tmp_path / "work")
            DistillationPipeline(config).run(resume=False)

            doc = (config.workdir / "investigator/architecture_overview.md").read_text(encoding="utf-8")
            self.assertIn("## Model Analysis", doc)

    def test_hybrid_retriever_finds_symbols_after_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            repo = tmp_path / "repo"
            repo.mkdir()
            _write_sample_repo(repo)
            config = PipelineConfig.default(repo, tmp_path / "work")
            DistillationPipeline(config).run(resume=False)

            hits = HybridRetriever(config.workdir / "index").search("Greeter greet", top_k=2)

            self.assertTrue(hits)
            self.assertTrue(hits[0].chunk.path.endswith("Greeter.java"))
            self.assertGreater(hits[0].score, 0)
            self.assertIn("greet", hits[0].chunk.text)


if __name__ == "__main__":
    unittest.main()
