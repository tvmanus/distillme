"""Safe read-only CLI execution for agentic repository exploration.

The executor validates every command against an approved whitelist before
invoking it as a subprocess.  Mutation commands are unconditionally rejected.
No shell interpretation is used; arguments are passed directly to the OS.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

# Commands that are unconditionally forbidden regardless of arguments.
_FORBIDDEN: frozenset[str] = frozenset(
    {
        "rm",
        "mv",
        "cp",
        "chmod",
        "chown",
        "sudo",
        "curl",
        "wget",
        "ssh",
        "nc",
        "ncat",
        "netcat",
        "pip",
        "pip3",
        "npm",
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "brew",
        "make",
        "mvn",
        "gradle",
        "python",
        "python3",
        "ruby",
        "node",
        "sh",
        "bash",
        "zsh",
        "eval",
        "exec",
        "source",
        "env",
        "export",
    }
)

# Approved first-token commands.  ``git`` has a further sub-command allowlist.
_APPROVED_COMMANDS: frozenset[str] = frozenset(
    {
        "ls",
        "find",
        "tree",
        "fd",
        "locate",
        "grep",
        "rg",
        "ag",
        "ack",
        "cat",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "cut",
        "sed",
        "awk",
        "git",
        "jq",
        "yq",
        "xmlstarlet",
        "ctags",
    }
)

# Only these git sub-commands are permitted.
_APPROVED_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "log",
        "blame",
        "grep",
        "ls-files",
        "show",
        "diff",
        "shortlog",
        "tag",
    }
)


@dataclasses.dataclass(frozen=True)
class CliResult:
    """Output from a single safe CLI command execution."""

    command: str
    stdout: str
    stderr: str
    returncode: int
    truncated: bool

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    def summary(self, max_lines: int = 8) -> str:
        """Return a short human-readable summary of the output."""
        if not self.stdout.strip():
            reason = self.stderr.strip() or "no output"
            return f"(exit {self.returncode}: {reason})"
        lines = self.stdout.strip().splitlines()
        kept = lines[:max_lines]
        suffix = f"\n  … {len(lines) - max_lines} more lines" if len(lines) > max_lines else ""
        return "\n".join(f"  {ln}" for ln in kept) + suffix


class CliExecutor:
    """Executes a curated set of read-only CLI tools against a repository path.

    Commands are validated against :data:`_APPROVED_COMMANDS` before execution.
    The working directory is always set to *repo_path* so that relative paths
    in arguments resolve correctly.

    All output is captured and optionally truncated to *max_output_chars*.
    The executor never raises on command failure — callers inspect
    :attr:`CliResult.returncode` instead.
    """

    def __init__(
        self,
        repo_path: Path,
        timeout: int = 15,
        max_output_chars: int = 8192,
    ) -> None:
        self.repo_path = repo_path
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def run(self, args: list[str]) -> CliResult:
        """Validate and execute *args*, returning a :class:`CliResult`.

        Raises :class:`ValueError` when the command is forbidden or not in the
        approved list.  Subprocess failures (non-zero exit, timeout, missing
        binary) are returned as :class:`CliResult` values rather than
        exceptions.
        """
        if not args:
            raise ValueError("empty command")
        cmd = args[0]
        if cmd in _FORBIDDEN:
            raise ValueError(f"command is forbidden: {cmd!r}")
        if cmd not in _APPROVED_COMMANDS:
            raise ValueError(f"command not in approved list: {cmd!r}")
        if cmd == "git":
            sub = args[1] if len(args) > 1 else "(none)"
            if sub not in _APPROVED_GIT_SUBCOMMANDS:
                raise ValueError(f"git sub-command not approved: {sub!r}")
        try:
            proc = subprocess.run(
                args,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            raw = proc.stdout
            truncated = len(raw) > self.max_output_chars
            return CliResult(
                command=" ".join(args),
                stdout=raw[: self.max_output_chars],
                stderr=proc.stderr[:512],
                returncode=proc.returncode,
                truncated=truncated,
            )
        except subprocess.TimeoutExpired:
            return CliResult(
                command=" ".join(args),
                stdout="",
                stderr="command timed out",
                returncode=-1,
                truncated=False,
            )
        except FileNotFoundError:
            return CliResult(
                command=" ".join(args),
                stdout="",
                stderr=f"command not found: {args[0]!r}",
                returncode=-1,
                truncated=False,
            )
