"""Embedding model connector for vector search.

Provides a pluggable :class:`EmbeddingClient` interface with three concrete
implementations:

* :class:`StubEmbeddingClient` — deterministic hash-projection vectors,
  works offline with no additional dependencies; used for tests and when no
  embedding endpoint is configured.

* :class:`HttpEmbeddingClient` — calls any server that exposes
  ``POST /v1/embeddings`` (OpenAI-compatible format), such as Ollama,
  text-embeddings-inference, vLLM, or any managed provider (OpenAI,
  Mistral, Cohere, etc.).

Use :func:`make_embedding_client` to obtain the appropriate client based on
the configured endpoint string.
"""

from __future__ import annotations

import abc
import hashlib
import json
import math
import urllib.error
import urllib.request
from typing import Sequence


class EmbeddingClient(abc.ABC):
    """Abstract interface for producing dense vector representations of text.

    Implement this interface to plug any embedding backend — local ONNX
    models, remote API providers, or sentence-transformers — into the
    distillation retrieval layer.  The embedding model is not subject to
    the single-LLM exclusivity constraint and may be called at any time
    during pipeline execution.
    """

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one normalised embedding vector per input text.

        Parameters
        ----------
        texts:
            One or more strings to embed.  The implementation must return a
            list of the same length, where each element is a ``list[float]``
            of consistent dimension.
        """


class StubEmbeddingClient(EmbeddingClient):
    """Deterministic hash-projection embedding client requiring no external deps.

    Uses a signed random-projection trick: each token is hashed with SHA-256
    and its sign is projected into a fixed-dimension vector that is then
    L2-normalised.  The resulting vectors are semantically meaningless but
    deterministic and suitable for offline use, unit tests, and as a
    fallback when no embedding endpoint is configured.

    Parameters
    ----------
    dimensions:
        Length of the produced embedding vectors (default: 384).
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_embed(text, self.dimensions) for text in texts]


class HttpEmbeddingClient(EmbeddingClient):
    """OpenAI-compatible HTTP embedding client.

    Connects to any server that exposes ``POST /v1/embeddings`` with an
    OpenAI-compatible request/response body, for example:

    * **Ollama**: ``http://localhost:11434`` with model ``nomic-embed-text``
    * **text-embeddings-inference**: ``http://localhost:8080``
    * **vLLM**: ``http://localhost:8000`` with an embeddings model
    * **OpenAI API**: ``https://api.openai.com`` with model
      ``text-embedding-3-small``
    * **Mistral API**: ``https://api.mistral.ai`` with model
      ``mistral-embed``

    Authentication headers (e.g. ``Authorization: Bearer <key>``) should be
    added via a proxy or by subclassing and overriding :meth:`_headers`.

    Parameters
    ----------
    endpoint:
        Base URL of the embedding server (without trailing slash).
    model:
        Model identifier forwarded in the request body.
    timeout:
        Request timeout in seconds (default: 60).
    api_key:
        Optional bearer token added to the ``Authorization`` header.
    """

    def __init__(
        self,
        endpoint: str,
        model: str = "text-embedding-ada-002",
        timeout: int = 60,
        api_key: str = "",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._api_key = api_key

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = json.dumps({"input": list(texts), "model": self.model}).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self.endpoint}/v1/embeddings",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read())
            return [item["embedding"] for item in body["data"]]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Embedding inference failed for endpoint {self.endpoint}: {exc}"
            ) from exc


def make_embedding_client(endpoint: str = "local", model: str = "local", api_key: str = "") -> EmbeddingClient:
    """Return an :class:`EmbeddingClient` appropriate for *endpoint*.

    Returns an :class:`HttpEmbeddingClient` when *endpoint* begins with
    ``http://`` or ``https://``; otherwise returns the deterministic
    :class:`StubEmbeddingClient` for offline use and testing.

    Parameters
    ----------
    endpoint:
        Embedding server base URL or ``"local"`` for the offline stub.
    model:
        Model name to use with the HTTP backend (ignored by stub).
    api_key:
        Optional bearer token for the HTTP backend.
    """
    if endpoint.startswith(("http://", "https://")):
        return HttpEmbeddingClient(endpoint=endpoint, model=model, api_key=api_key)
    return StubEmbeddingClient()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hash_embed(text: str, dimensions: int = 384) -> list[float]:
    """Return a deterministic L2-normalised hash-projection vector for *text*.

    Splits the input on whitespace, lower-cases each token, and projects
    them into *dimensions*-dimensional space using SHA-256 sign-projection.
    The vector is L2-normalised before return.
    """
    tokens = [token.lower() for token in text.split() if token]
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
