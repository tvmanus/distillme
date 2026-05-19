"""Structured trace, prompt, retrieval, and metric logging."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class TraceLogger:
    """Append-only JSONL trace logger with run lineage metadata."""

    def __init__(self, log_dir: Path, run_id: str | None = None) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or str(uuid.uuid4())
        self.path = self.log_dir / "trace.jsonl"

    def event(self, name: str, **payload: Any) -> None:
        record = {
            "run_id": self.run_id,
            "timestamp": time.time(),
            "event": name,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
