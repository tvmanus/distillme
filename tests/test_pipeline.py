from __future__ import annotations

import json
import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distillme.cli_tools import CliExecutor, CliResult
from distillme.config import PipelineConfig
from distillme.inference import HttpLLMClient, LLMClient, StubLLMClient, make_client
from distillme.orchestration import STAGES, DistillationPipeline
from distillme.retrieval import HybridRetriever
from distillme.schemas import InvestigationTrace, ModelSpec
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


class CliToolsTests(unittest.TestCase):
    def test_executor_rejects_forbidden_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = CliExecutor(Path(directory))
            with self.assertRaises(ValueError):
                executor.run(["rm", "-rf", "."])

    def test_executor_rejects_unapproved_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = CliExecutor(Path(directory))
            with self.assertRaises(ValueError):
                executor.run(["curl", "http://example.com"])

    def test_executor_rejects_unapproved_git_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = CliExecutor(Path(directory))
            with self.assertRaises(ValueError):
                executor.run(["git", "push"])

    def test_executor_runs_find_in_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "Hello.java").write_text("class Hello {}\n")
            executor = CliExecutor(tmp)
            result = executor.run(["find", ".", "-name", "*.java", "-type", "f"])
            self.assertIsInstance(result, CliResult)
            self.assertEqual(result.returncode, 0)
            self.assertIn("Hello.java", result.stdout)

    def test_executor_runs_grep_in_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "Hello.java").write_text("class Hello {}\n")
            executor = CliExecutor(tmp)
            result = executor.run(["grep", "-r", "class", ".", "--include=*.java"])
            self.assertEqual(result.returncode, 0)
            self.assertIn("Hello", result.stdout)

    def test_cli_result_summary_truncates_long_output(self) -> None:
        result = CliResult(
            command="find . -type f",
            stdout="\n".join(f"./file{i}.java" for i in range(20)),
            stderr="",
            returncode=0,
            truncated=False,
        )
        summary = result.summary(max_lines=5)
        self.assertIn("file0", summary)
        self.assertIn("more lines", summary)

    def test_investigation_trace_renders_to_markdown(self) -> None:
        trace = InvestigationTrace(
            objective="Find auth patterns",
            hypothesis="Auth is handled in a dedicated service",
            known_evidence=("AuthService.java found",),
            uncertainties=("Runtime behaviour unverified",),
            commands_run=("grep -r Auth .",),
            command_summaries=("Matched 3 files",),
            updated_understanding="Auth layer confirmed via grep evidence.",
            next_investigation_step="Inspect AuthService.java internals.",
            confidence=0.72,
        )
        md = trace.to_markdown()
        self.assertIn("OBJECTIVE", md)
        self.assertIn("CURRENT HYPOTHESIS", md)
        self.assertIn("KNOWN EVIDENCE", md)
        self.assertIn("COMMANDS RUN", md)
        self.assertIn("0.72", md)

    def test_agentic_investigator_loop_runs_against_sample_repo(self) -> None:
        from distillme.investigator import AgenticInvestigatorLoop
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            _write_sample_repo(tmp)
            executor = CliExecutor(tmp)
            loop = AgenticInvestigatorLoop(executor)
            trace = loop.investigate("architecture_overview.md", "architecture package", 0.5)
            self.assertIsInstance(trace, InvestigationTrace)
            self.assertGreater(len(trace.commands_run), 0)

    def test_dataset_records_include_investigation_trace_for_agentic_categories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            repo = tmp_path / "repo"
            repo.mkdir()
            _write_sample_repo(repo)
            config = PipelineConfig.default(repo, tmp_path / "work")
            DistillationPipeline(config).run(resume=False)

            dataset_path = config.workdir / "dataset/instruction_dataset.jsonl"
            records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
            agentic = [r for r in records if r["task_category"] in {"agentic_task", "cli_exploration_task"}]
            self.assertTrue(agentic, "expected agentic/cli_exploration records")
            for record in agentic:
                self.assertIn("investigation_trace", record)

    def test_new_task_categories_present_in_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            repo = tmp_path / "repo"
            repo.mkdir()
            _write_sample_repo(repo)
            config = PipelineConfig.default(repo, tmp_path / "work")
            DistillationPipeline(config).run(resume=False)

            dataset_path = config.workdir / "dataset/instruction_dataset.jsonl"
            records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
            categories = {r["task_category"] for r in records}
            self.assertIn("cli_exploration_task", categories)
            self.assertIn("multi_hop_navigation_task", categories)


if __name__ == "__main__":
    unittest.main()
