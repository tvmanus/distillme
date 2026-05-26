"""Pluggable LLM inference interface with stub and HTTP backends."""

from __future__ import annotations

import abc
import json
import logging
import threading
import urllib.error
import urllib.request

from distillme.schemas import ModelSpec

_LOG = logging.getLogger(__name__)

# Qwen3 / DeepSeek-R1 inline thinking delimiters.
# Ollama ≥ 0.9 returns thinking in a separate ``reasoning`` field, but older
# builds and raw HuggingFace serving embed the CoT inside ``content``.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

# Module-level lock shared by all :class:`ExclusiveLLMClient` instances.
# Ensures that at most one LLM ``generate()`` call is executing at any time
# across the entire pipeline process.  Embedding clients are exempt from
# this constraint and may be called concurrently.
_PIPELINE_LLM_LOCK = threading.Lock()


class LLMClient(abc.ABC):
    """Abstract LLM inference backend.

    Implement this interface to connect any model family (Gemini, Claude,
    Qwen, Llama, Mistral, …) to the distillation pipeline.  The pipeline
    calls :meth:`generate` for every document or dataset-example turn that
    requires model output.
    """

    @abc.abstractmethod
    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Return a text completion for the given system and user turn."""


class StubLLMClient(LLMClient):
    """Deterministic offline stub used in tests and when no endpoint is configured.

    Returns a clearly-labelled placeholder that preserves the uncertainty
    guardrail and directs operators to configure a real endpoint.
    """

    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:  # noqa: ARG002
        label = f"{self.model_spec.family}/{self.model_spec.model}"
        return (
            f"[stub:{label}] Real generation requires a configured HTTP inference endpoint. "
            "All claims must be validated against retrieved source evidence before training use. "
            "Uncertain about any aspect not directly observable in the indexed artifacts."
        )


class HttpLLMClient(LLMClient):
    """OpenAI-compatible HTTP client for local or distributed model serving.

    Connects to any inference server that exposes ``POST /v1/chat/completions``
    (e.g. vLLM, Ollama, LM Studio, Tabby, together.ai, Fireworks, etc.).

    Extended-thinking aware
    -----------------------
    Qwen3 and DeepSeek-R1 models produce a chain-of-thought (CoT) before the
    final answer.  Ollama ≥ 0.9 surfaces this as a separate ``reasoning`` field
    while putting the answer in ``content``.  Older builds or raw HuggingFace
    servers embed the CoT inline as ``<think>…</think>`` inside ``content``.

    :meth:`_extract_content` normalises both forms so callers always receive
    the final answer only, never the raw thinking tokens.
    """

    def __init__(self, model_spec: ModelSpec, timeout: int = 1800) -> None:
        self.model_spec = model_spec
        # 30-minute default: thinking-mode models generating 8 000–12 000 tokens
        # on a local GPU can take several minutes per call.
        self.timeout = timeout
        self._base = model_spec.endpoint.rstrip("/")

    # ------------------------------------------------------------------
    # Thinking-strip parser
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(message: dict, finish_reason: str | None) -> str:
        """Return the final answer from a chat completion message dict.

        Handles two extended-thinking response shapes:

        **Separated** (Ollama ≥ 0.9 with PARSER qwen3.5 / DeepSeek API):
            The server already split CoT into ``reasoning`` (or
            ``reasoning_content``) and the final answer into ``content``.
            We return ``content`` directly — do *not* apply inline
            ``<think>`` stripping here, because that would corrupt any
            answer that legitimately contains the literal text ``<think>``
            (e.g. the model explaining its own format).

        **Inline** (older Ollama builds, raw HuggingFace / vLLM serving):
            ``content`` contains ``<think>…</think>`` followed by the answer.
            We locate the *last* ``</think>`` tag and return everything after it.
            If the closing tag is absent the model was truncated mid-think and
            the answer was never written — we return ``""`` so the caller's
            empty-response guard fires.

        A ``finish_reason == "length"`` warning is emitted in both truncation
        scenarios so operators know to increase ``max_tokens``.
        """
        # ── Separated form ────────────────────────────────────────────────
        # Presence of a ``reasoning`` or ``reasoning_content`` key signals
        # that the server already stripped thinking from content.  Trust it.
        if "reasoning" in message or "reasoning_content" in message:
            content: str = message.get("content") or ""
            if finish_reason == "length":
                _LOG.warning(
                    "Generation truncated (finish_reason=length) — answer may be incomplete."
                )
            return content

        # ── Inline form ───────────────────────────────────────────────────
        content = message.get("content") or ""

        if _THINK_OPEN in content:
            last_close = content.rfind(_THINK_CLOSE)
            if last_close != -1:
                # Thinking completed normally — answer follows the closing tag.
                content = content[last_close + len(_THINK_CLOSE):].lstrip("\n").strip()
            else:
                # Truncated inside the thinking block — no answer was produced.
                if finish_reason == "length":
                    _LOG.warning(
                        "Generation truncated (finish_reason=length) inside <think> block "
                        "— answer is empty. Increase max_tokens."
                    )
                content = ""
        elif finish_reason == "length":
            _LOG.warning(
                "Generation truncated (finish_reason=length) — answer may be incomplete."
            )

        return content

    # ------------------------------------------------------------------
    # Core generate
    # ------------------------------------------------------------------

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        payload = json.dumps(
            {
                "model": self.model_spec.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                # Ollama ignores the model-file num_ctx default (2048 tokens) unless
                # told explicitly via options.num_ctx.  Non-Ollama OpenAI-compatible
                # servers silently ignore the options field.
                "options": {"num_ctx": self.model_spec.max_context_tokens},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._base}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read())
            choice = body["choices"][0]
            finish_reason: str | None = choice.get("finish_reason")
            content = self._extract_content(choice["message"], finish_reason)
            usage = body.get("usage", {})
            if usage:
                _LOG.debug(
                    "Token usage — prompt: %d  completion: %d  finish: %s",
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    finish_reason,
                )
            return content
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"LLM inference failed for endpoint {self._base}: {exc}"
            ) from exc


def make_client(spec: ModelSpec) -> LLMClient:
    """Return an :class:`LLMClient` appropriate for *spec*.

    Returns an :class:`HttpLLMClient` when ``spec.endpoint`` begins with
    ``http://`` or ``https://``; otherwise returns the deterministic
    :class:`StubLLMClient` for offline use and testing.
    """
    if spec.endpoint.startswith(("http://", "https://")):
        return HttpLLMClient(spec)
    return StubLLMClient(spec)


class ExclusiveLLMClient(LLMClient):
    """Wraps any :class:`LLMClient` to enforce single-LLM-at-a-time execution.

    Acquires :data:`_PIPELINE_LLM_LOCK` before delegating to the wrapped
    client and releases it after the call returns.  This guarantees that no
    two LLM ``generate()`` calls overlap within the same process, regardless
    of threading.

    Embedding clients (:class:`~distillme.embedding.EmbeddingClient`) are
    exempt from this lock and may be called at any time.

    Parameters
    ----------
    inner:
        The underlying :class:`LLMClient` to delegate to.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        with _PIPELINE_LLM_LOCK:
            return self._inner.generate(system, user, max_tokens)


def make_exclusive_client(spec: ModelSpec) -> ExclusiveLLMClient:
    """Return an :class:`ExclusiveLLMClient` wrapping the client for *spec*.

    Convenience wrapper combining :func:`make_client` with
    :class:`ExclusiveLLMClient` so all pipeline LLM calls automatically
    respect the single-LLM exclusivity guarantee.
    """
    return ExclusiveLLMClient(make_client(spec))
