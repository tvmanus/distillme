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
        max_output_chars: int = 32_768,
    ) -> None:
        self.repo_path = repo_path
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def run(self, args: list[str]) -> CliResult:
        """Validate and execute *args*, returning a :class:`CliResult`.

        In addition to the OS-level approved commands, the following virtual
        compound commands are supported and dispatched before the whitelist check:

        ``grep_context PATTERN TARGET [CONTEXT_LINES]``
            Finds *PATTERN* (extended regex) in *TARGET* (file or directory)
            and returns *CONTEXT_LINES* lines of source around each match.
            Internally chains ``grep -En`` → ``sed -n 'START,ENDp'`` for every
            distinct match location (up to 5 blocks).  Output blocks are
            prefixed with ``=== filepath:lineno ===`` headers so the LLM can
            cite exact locations.

        Raises :class:`ValueError` when the command is forbidden or not in the
        approved list.  Subprocess failures (non-zero exit, timeout, missing
        binary) are returned as :class:`CliResult` values rather than
        exceptions.
        """
        if not args:
            raise ValueError("empty command")
        cmd = args[0]
        # ── Virtual compound commands (dispatched before the OS whitelist) ────
        if cmd == "grep_context":
            return self._run_grep_context(args)
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

    # ------------------------------------------------------------------
    # Compound tool: grep_context
    # ------------------------------------------------------------------

    def _run_grep_context(self, args: list[str]) -> CliResult:
        """Implement the ``grep_context`` virtual compound command.

        Algorithm
        ---------
        1. Run ``grep -En`` (extended regex, with line numbers) against *target*.
           Adds ``-r --include=<source exts>`` when *target* is a directory.
        2. Parse ``(filepath, lineno)`` pairs from grep output.
        3. Collapse overlapping match locations into distinct windows.
        4. For each window run ``sed -n 'START,ENDp' filepath`` to extract the
           code block.  Windows are biased 1:2 before/after the match line so
           the model sees the context that follows each hit.
        5. Return up to 5 blocks separated by ``=== filepath:lineno ===`` headers.

        Parameters decoded from *args*
        --------------------------------
        args[1]  PATTERN       Extended-regex pattern.  Use ``|`` for alternation.
        args[2]  TARGET        File or directory path (relative to repo root).
        args[3]  CONTEXT_LINES Number of lines per block (default 40).
        """
        if len(args) < 3:
            raise ValueError("grep_context requires: PATTERN TARGET [CONTEXT_LINES]")

        pattern, target = args[1], args[2]
        context_lines = int(args[3]) if len(args) > 3 else 40

        # Resolve target to decide file vs. directory mode.
        target_abs = (self.repo_path / target.lstrip("./")).resolve()
        is_dir = target_abs.is_dir()

        # Step 1 — grep with extended regex and line numbers.
        grep_cmd = ["grep", "-E", "-n"]
        if is_dir:
            grep_cmd += [
                "-r",
                "--include=*.java", "--include=*.py", "--include=*.kt",
                "--include=*.go", "--include=*.ts", "--include=*.js",
                "--include=*.scala", "--include=*.cs",
            ]
        grep_cmd += [pattern, target]

        grep_res = self.run(grep_cmd)
        if not grep_res.stdout.strip():
            return CliResult(
                command=" ".join(args),
                stdout="",
                stderr=grep_res.stderr or f"no matches for {pattern!r} in {target}",
                returncode=1,
                truncated=False,
            )

        # Step 2 — parse (filepath, lineno) pairs.
        matches: list[tuple[str, int]] = []
        for line in grep_res.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("Binary"):
                continue
            if is_dir:
                # format: ./path/File.java:42:content
                parts = line.split(":", 2)
                if len(parts) >= 2:
                    try:
                        matches.append((parts[0], int(parts[1])))
                    except ValueError:
                        pass
            else:
                # format: 42:content
                parts = line.split(":", 1)
                try:
                    matches.append((target, int(parts[0])))
                except (ValueError, IndexError):
                    pass

        if not matches:
            return CliResult(
                command=" ".join(args),
                stdout="",
                stderr="could not parse grep output",
                returncode=1,
                truncated=False,
            )

        # Step 3 — collapse overlapping windows and extract blocks via sed.
        matches.sort()  # sort by (file, lineno) so nearby hits cluster
        blocks: list[str] = []
        total_chars = 0
        seen: set[tuple[str, int]] = set()  # (filepath, coarse_region)

        for filepath, lineno in matches:
            # Coarsen to regions so nearby matches share one block.
            region = (filepath, lineno // max(1, context_lines))
            if region in seen:
                continue
            seen.add(region)

            # Bias window: show ~1/3 before, ~2/3 after the match.
            start = max(1, lineno - context_lines // 3)
            end = lineno + (2 * context_lines // 3)

            sed_res = self.run(["sed", "-n", f"{start},{end}p", filepath])
            if sed_res.succeeded and sed_res.stdout.strip():
                block = f"=== {filepath}:{lineno} ===\n{sed_res.stdout.rstrip()}\n"
                blocks.append(block)
                total_chars += len(block)

            if len(blocks) >= 5 or total_chars >= self.max_output_chars:
                break

        if not blocks:
            return CliResult(
                command=" ".join(args),
                stdout="",
                stderr="sed extraction yielded no content",
                returncode=1,
                truncated=False,
            )

        combined = "\n\n".join(blocks)
        return CliResult(
            command=" ".join(args),
            stdout=combined[: self.max_output_chars],
            stderr="",
            returncode=0,
            truncated=len(combined) > self.max_output_chars,
        )
