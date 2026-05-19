"""Pluggable LLM inference interface with stub and HTTP backends."""

from __future__ import annotations

import abc
import json
import urllib.error
import urllib.request

from distillme.schemas import ModelSpec


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
    """

    def __init__(self, model_spec: ModelSpec, timeout: int = 120) -> None:
        self.model_spec = model_spec
        self.timeout = timeout
        self._base = model_spec.endpoint.rstrip("/")

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
            return body["choices"][0]["message"]["content"]
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
