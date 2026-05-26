"""Command-line interface for distillme."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from distillme.config import PipelineConfig
from distillme.orchestration import DistillationPipeline, STAGES


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Run the distillme knowledge distillation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="write a default JSON configuration")
    init.add_argument("--repository", required=True, type=Path)
    init.add_argument("--workdir", required=True, type=Path)
    init.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run", help="run the pipeline")
    run.add_argument("--config", required=True, type=Path)
    resume_group = run.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore existing state and run all stages from scratch",
    )
    resume_group.add_argument(
        "--from-stage",
        choices=STAGES,
        metavar="STAGE",
        dest="from_stage",
        help=(
            "preserve outputs from completed earlier stages and re-run from STAGE onward "
            f"(valid values: {', '.join(STAGES)})"
        ),
    )

    args = parser.parse_args(argv)
    if args.command == "init":
        config = PipelineConfig.default(args.repository.resolve(), args.workdir.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(config.to_jsonable(), indent=2), encoding="utf-8")
        return 0
    if args.command == "run":
        config = PipelineConfig.from_file(args.config)
        results = DistillationPipeline(config).run(
            resume=not args.no_resume,
            from_stage=getattr(args, "from_stage", None),
        )
        print(json.dumps({stage: result.status for stage, result in results.items()}, indent=2))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
