"""Command-line interface for distillme."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distillme.config import PipelineConfig
from distillme.orchestration import DistillationPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the distillme knowledge distillation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="write a default JSON configuration")
    init.add_argument("--repository", required=True, type=Path)
    init.add_argument("--workdir", required=True, type=Path)
    init.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run", help="run the pipeline")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--no-resume", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "init":
        config = PipelineConfig.default(args.repository.resolve(), args.workdir.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(config.to_jsonable(), indent=2), encoding="utf-8")
        return 0
    if args.command == "run":
        config = PipelineConfig.from_file(args.config)
        results = DistillationPipeline(config).run(resume=not args.no_resume)
        print(json.dumps({stage: result.status for stage, result in results.items()}, indent=2))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
