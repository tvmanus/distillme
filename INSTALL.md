# Installation and Configuration Manual

This document covers everything needed to install **distillme**, connect it to
external resources (LLM API providers, embedding models, vector databases),
and run the full distillation pipeline.

---

## Table of contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Optional extras](#3-optional-extras)
4. [Configuration file reference](#4-configuration-file-reference)
5. [LLM provider configuration](#5-llm-provider-configuration)
   - [OpenAI](#51-openai)
   - [Anthropic Claude](#52-anthropic-claude)
   - [Google Gemini via OpenAI-compatible proxy](#53-google-gemini-via-openai-compatible-proxy)
   - [Ollama (local models)](#54-ollama-local-models)
   - [vLLM / LM Studio / any OpenAI-compatible server](#55-vllm--lm-studio--any-openai-compatible-server)
6. [Embedding model configuration](#6-embedding-model-configuration)
   - [Stub (default, offline)](#61-stub-default-offline)
   - [Ollama embedding models](#62-ollama-embedding-models)
   - [OpenAI / Mistral / Cohere embeddings](#63-openai--mistral--cohere-embeddings)
   - [text-embeddings-inference (Hugging Face)](#64-text-embeddings-inference-hugging-face)
7. [Vector database configuration](#7-vector-database-configuration)
   - [Local JSONL (default, no extra deps)](#71-local-jsonl-default-no-extra-deps)
   - [ChromaDB (pip-installable persistent vector store)](#72-chromadb-pip-installable-persistent-vector-store)
8. [Single-LLM execution guarantee](#8-single-llm-execution-guarantee)
9. [Memory / MCP server integration](#9-memory--mcp-server-integration)
10. [Running the pipeline](#10-running-the-pipeline)
11. [Docker](#11-docker)
12. [Complete configuration examples](#12-complete-configuration-examples)
13. [Environment variable reference](#13-environment-variable-reference)

---

## 1. Requirements

| Requirement | Minimum version |
|---|---|
| Python | 3.11 |
| pip | 23+ |
| Git | Any recent version |

The core package uses **only Python standard library** modules.  No external
Python packages are required unless you opt into the Chroma vector backend
(see [§3](#3-optional-extras)).

---

## 2. Installation

### From source (recommended during development)

```bash
git clone https://github.com/tvmanus/distillme.git
cd distillme
pip install -e .
```

### From PyPI (once published)

```bash
pip install distillme
```

Verify the installation:

```bash
distillme --help
```

---

## 3. Optional extras

### ChromaDB vector store

Enables `vector_backend = "chroma"` for persistent, embedding-based retrieval:

```bash
pip install 'distillme[chroma]'
```

This pulls in `chromadb >= 1.0.0`.

### sentence-transformers (local neural embeddings)

Enables local neural embedding models without a network call.  Use with an
Ollama or text-embeddings-inference server that serves a
sentence-transformers-compatible model:

```bash
pip install 'distillme[sentence-transformers]'
```

### All extras at once

```bash
pip install 'distillme[all]'
```

---

## 4. Configuration file reference

Generate a default configuration file:

```bash
distillme init \
  --repository /path/to/your/java/repository \
  --workdir    /path/to/distillme-workdir \
  --output     config.json
```

The generated file has the following structure:

```json
{
  "repository_path": "/absolute/path/to/java/repository",
  "workdir": "/absolute/path/to/distillme-workdir",
  "models": {
    "investigator": {
      "family": "gemini",
      "model": "gemini-1.5-pro",
      "endpoint": "http://localhost:8080",
      "max_context_tokens": 1000000,
      "batch_size": 1
    },
    "teacher": {
      "family": "claude",
      "model": "claude-3-5-sonnet-20241022",
      "endpoint": "http://localhost:8081",
      "max_context_tokens": 200000,
      "batch_size": 1
    },
    "student": {
      "family": "qwen2.5",
      "model": "Qwen2.5-Coder-7B-Instruct",
      "endpoint": "local",
      "max_context_tokens": 32768,
      "batch_size": 8
    }
  },
  "retrieval": {
    "vector_backend": "local-jsonl",
    "embedding_endpoint": "local",
    "embedding_model": "local",
    "embedding_api_key": "",
    "dense_weight": 0.45,
    "sparse_weight": 0.35,
    "symbol_weight": 0.20,
    "max_chunk_lines": 120,
    "top_k": 8
  },
  "include_globs": ["**/*"],
  "exclude_dirs": [
    ".git", ".gradle", ".idea", ".mvn/wrapper",
    "build", "target", "node_modules", "dist", "__pycache__"
  ]
}
```

### Key rules

* **`models.investigator.family`**, **`models.teacher.family`**, and
  **`models.student.family`** must be three **distinct** strings.  The
  validation step rejects configurations where any two roles share a family,
  enforcing the heterogeneous-LLM distillation guarantee.

* Set `endpoint` to `"local"` (or any non-HTTP string) to use the offline stub
  client.  Set it to an `http://` or `https://` URL to use the real model.

* `retrieval.vector_backend` controls which retrieval backend is used:
  `"local-jsonl"` (default, no extra deps) or `"chroma"` (requires ChromaDB).

---

## 5. LLM provider configuration

The pipeline makes OpenAI-compatible `POST /v1/chat/completions` requests.
Any server or proxy that speaks this API works.

### 5.1 OpenAI

```json
"investigator": {
  "family": "openai",
  "model": "gpt-4o",
  "endpoint": "https://api.openai.com",
  "max_context_tokens": 128000,
  "batch_size": 1
}
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

> **Note**: The built-in `HttpLLMClient` does not yet inject API keys
> automatically.  Use a local proxy such as
> [LiteLLM](https://github.com/BerriAI/litellm) or pass the key via a
> custom subclass.  Alternatively, point `endpoint` at LiteLLM which handles
> authentication transparently.

Using **LiteLLM** as a transparent proxy (recommended):

```bash
pip install litellm
OPENAI_API_KEY=sk-... litellm --model gpt-4o --port 8080
```

Then set `"endpoint": "http://localhost:8080"` in the config.

### 5.2 Anthropic Claude

Claude does not natively expose an OpenAI-compatible endpoint, but LiteLLM
provides a drop-in proxy:

```bash
pip install litellm
ANTHROPIC_API_KEY=sk-ant-... litellm --model claude-3-5-sonnet-20241022 --port 8081
```

Config:

```json
"teacher": {
  "family": "claude",
  "model": "claude-3-5-sonnet-20241022",
  "endpoint": "http://localhost:8081",
  "max_context_tokens": 200000,
  "batch_size": 1
}
```

### 5.3 Google Gemini via OpenAI-compatible proxy

```bash
pip install litellm
GOOGLE_API_KEY=AIza... litellm --model gemini/gemini-1.5-pro --port 8082
```

Config:

```json
"investigator": {
  "family": "gemini",
  "model": "gemini-1.5-pro",
  "endpoint": "http://localhost:8082",
  "max_context_tokens": 1000000,
  "batch_size": 1
}
```

### 5.4 Ollama (local models)

[Ollama](https://ollama.com) exposes an OpenAI-compatible endpoint at
`http://localhost:11434`.

```bash
ollama serve
ollama pull llama3.3
ollama pull qwen2.5-coder:7b
```

Config:

```json
"investigator": {
  "family": "llama",
  "model": "llama3.3",
  "endpoint": "http://localhost:11434",
  "max_context_tokens": 128000,
  "batch_size": 1
},
"teacher": {
  "family": "mistral",
  "model": "mistral-nemo",
  "endpoint": "http://localhost:11434",
  "max_context_tokens": 128000,
  "batch_size": 1
}
```

> **Important**: Each model role must use a distinct `family` string even when
> both point at the same Ollama instance.

### 5.5 vLLM / LM Studio / any OpenAI-compatible server

Set `endpoint` to the base URL of your server and `model` to the model
identifier the server expects:

```json
"endpoint": "http://localhost:8000",
"model": "meta-llama/Meta-Llama-3.1-70B-Instruct"
```

---

## 6. Embedding model configuration

Embedding configuration lives under `retrieval` in the config file.  Embedding
models are **not** subject to the single-LLM lock and may run concurrently with
other pipeline steps.

### 6.1 Stub (default, offline)

The default `embedding_endpoint = "local"` selects `StubEmbeddingClient`,
which uses deterministic hash-projection vectors.  No embedding server or
additional packages are required.

```json
"retrieval": {
  "embedding_endpoint": "local",
  "embedding_model": "local"
}
```

Use this when you do not need semantic embedding quality (e.g. testing,
development, or when the hybrid BM25+symbol retrieval is sufficient).

### 6.2 Ollama embedding models

```bash
ollama pull nomic-embed-text
ollama serve
```

Config:

```json
"retrieval": {
  "vector_backend": "chroma",
  "embedding_endpoint": "http://localhost:11434",
  "embedding_model": "nomic-embed-text"
}
```

### 6.3 OpenAI / Mistral / Cohere embeddings

Use LiteLLM as a unified proxy:

```bash
OPENAI_API_KEY=sk-... litellm --model text-embedding-3-small --port 8090
```

Config:

```json
"retrieval": {
  "vector_backend": "chroma",
  "embedding_endpoint": "http://localhost:8090",
  "embedding_model": "text-embedding-3-small"
}
```

For Mistral embeddings:

```bash
MISTRAL_API_KEY=... litellm --model mistral/mistral-embed --port 8091
```

For Cohere:

```bash
COHERE_API_KEY=... litellm --model cohere/embed-english-v3.0 --port 8092
```

For direct API access with an API key, set `embedding_api_key` in the config:

```json
"retrieval": {
  "vector_backend": "chroma",
  "embedding_endpoint": "https://api.openai.com",
  "embedding_model": "text-embedding-3-small",
  "embedding_api_key": "sk-..."
}
```

### 6.4 text-embeddings-inference (Hugging Face)

[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
(TEI) serves sentence-transformers models with an OpenAI-compatible API.

```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:turing-1.5 \
  --model-id BAAI/bge-large-en-v1.5
```

Config:

```json
"retrieval": {
  "vector_backend": "chroma",
  "embedding_endpoint": "http://localhost:8080",
  "embedding_model": "BAAI/bge-large-en-v1.5"
}
```

---

## 7. Vector database configuration

### 7.1 Local JSONL (default, no extra deps)

```json
"retrieval": {
  "vector_backend": "local-jsonl"
}
```

Uses `HybridRetriever`: a combination of deterministic dense hash-projection,
BM25 approximate sparse scoring, and symbol overlap.  Requires no additional
packages.  Best for rapid iteration, offline use, and repositories up to a
few thousand files.

**Tuning weights:**

```json
"retrieval": {
  "vector_backend": "local-jsonl",
  "dense_weight": 0.45,
  "sparse_weight": 0.35,
  "symbol_weight": 0.20,
  "top_k": 8
}
```

### 7.2 ChromaDB (pip-installable persistent vector store)

```bash
pip install 'distillme[chroma]'
```

```json
"retrieval": {
  "vector_backend": "chroma",
  "embedding_endpoint": "http://localhost:11434",
  "embedding_model": "nomic-embed-text",
  "top_k": 8
}
```

`ChromaRetriever`:

* Persists the vector index to `<workdir>/index/chroma/` using ChromaDB's
  built-in HNSW index.
* Indexes chunks lazily on the first `search()` call by reading the
  `chunks.jsonl` produced by the ingest stage.
* Subsequent pipeline runs skip re-indexing if the collection is already
  populated.
* Uses cosine similarity for nearest-neighbour search.

**Resetting the Chroma index** (e.g. after re-ingestion):

```bash
rm -rf /path/to/workdir/index/chroma/
```

---

## 8. Single-LLM execution guarantee

distillme enforces that **at most one LLM inference call is in-flight at any
given time** using a process-wide threading lock (`_PIPELINE_LLM_LOCK` in
`distillme.inference`).

All LLM clients used by the pipeline stages (investigator, teacher) are
automatically wrapped in `ExclusiveLLMClient`, which acquires this lock before
calling `generate()` and releases it on return.

**Embedding clients are exempt** from this lock.  The embedding model may be
called at any time, including during an ongoing LLM generation call (e.g. for
retrieval-augmented generation within a stage).

**Consequences:**

* Investigator and teacher LLM calls never overlap, even in a multi-threaded
  or asynchronous environment.
* You will never incur charges for two simultaneous long-context calls.
* The lock is process-scoped: running two separate `distillme run` processes
  against the same workdir does not provide cross-process mutual exclusion.

---

## 9. Memory / MCP server integration

distillme does not yet include a built-in MCP (Model Context Protocol) server.
To attach an external memory or tool server:

1. Start your MCP server (e.g.
   [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers))
   and expose it as an OpenAI-compatible endpoint.

2. Point a model role's `endpoint` at the MCP gateway:

   ```json
   "investigator": {
     "family": "mcp-gateway",
     "model": "gemini-1.5-pro",
     "endpoint": "http://localhost:9000"
   }
   ```

3. The MCP gateway is responsible for augmenting the request with memory
   context before forwarding to the underlying LLM.

Alternatively, inject memory context directly in the `_model_analysis` method
of `InvestigatorAgent` or the answer-generation logic in `TeacherAgent`.

---

## 10. Running the pipeline

### 1. Initialise a configuration

```bash
distillme init \
  --repository /path/to/java/repo \
  --workdir    /path/to/workdir \
  --output     config.json
```

Edit `config.json` to fill in real endpoints (see §5 and §6).

### 2. Run all stages

```bash
distillme run --config config.json
```

The pipeline is resumable by default.  Restart with `--no-resume` to re-run
from scratch:

```bash
distillme run --config config.json --no-resume
```

### 3. Check stage output

```
workdir/
├── index/                  # Artifact, chunk, graph JSONL + optional chroma/
├── investigator/           # Markdown findings (22 required documents)
├── dataset/
│   ├── instruction_dataset.jsonl
│   ├── manifest.json
│   └── validation_report.json
├── training/               # Student fine-tuning plan
├── evaluation/             # Benchmark definitions
├── logs/trace.jsonl        # Structured event trace
└── state.json              # Resumable stage state
```

---

## 11. Docker

Build and run with Docker:

```bash
docker build -t distillme .

docker run --rm \
  -v /path/to/java/repo:/repo:ro \
  -v /path/to/workdir:/workdir \
  -v /path/to/config.json:/config.json:ro \
  distillme run --config /config.json
```

To include the Chroma extra in the Docker image, add to the `Dockerfile`:

```dockerfile
RUN pip install --no-cache-dir -e '.[chroma]'
```

---

## 12. Complete configuration examples

### Fully local with Ollama

```json
{
  "repository_path": "/home/user/myproject",
  "workdir": "/home/user/distillme-out",
  "models": {
    "investigator": {
      "family": "llama",
      "model": "llama3.3",
      "endpoint": "http://localhost:11434",
      "max_context_tokens": 128000,
      "batch_size": 1
    },
    "teacher": {
      "family": "mistral",
      "model": "mistral-nemo",
      "endpoint": "http://localhost:11434",
      "max_context_tokens": 128000,
      "batch_size": 1
    },
    "student": {
      "family": "qwen2.5",
      "model": "Qwen2.5-Coder-7B-Instruct",
      "endpoint": "local",
      "max_context_tokens": 32768,
      "batch_size": 8
    }
  },
  "retrieval": {
    "vector_backend": "chroma",
    "embedding_endpoint": "http://localhost:11434",
    "embedding_model": "nomic-embed-text",
    "top_k": 8
  }
}
```

### Managed APIs (OpenAI + Anthropic + Google) via LiteLLM

```json
{
  "repository_path": "/home/user/myproject",
  "workdir": "/home/user/distillme-out",
  "models": {
    "investigator": {
      "family": "gemini",
      "model": "gemini-1.5-pro",
      "endpoint": "http://localhost:8082",
      "max_context_tokens": 1000000,
      "batch_size": 1
    },
    "teacher": {
      "family": "claude",
      "model": "claude-3-5-sonnet-20241022",
      "endpoint": "http://localhost:8081",
      "max_context_tokens": 200000,
      "batch_size": 1
    },
    "student": {
      "family": "qwen2.5",
      "model": "Qwen2.5-Coder-7B-Instruct",
      "endpoint": "local",
      "max_context_tokens": 32768,
      "batch_size": 8
    }
  },
  "retrieval": {
    "vector_backend": "chroma",
    "embedding_endpoint": "https://api.openai.com",
    "embedding_model": "text-embedding-3-small",
    "embedding_api_key": "sk-...",
    "top_k": 8
  }
}
```

Start LiteLLM proxies in separate terminals before running the pipeline:

```bash
GOOGLE_API_KEY=AIza...    litellm --model gemini/gemini-1.5-pro                --port 8082 &
ANTHROPIC_API_KEY=sk-ant-... litellm --model claude-3-5-sonnet-20241022    --port 8081 &
```

### Offline / testing (no LLM calls, no extra deps)

```json
{
  "repository_path": "/home/user/myproject",
  "workdir": "/home/user/distillme-out",
  "models": {
    "investigator": {
      "family": "gemini",
      "model": "gemini-stub",
      "endpoint": "local",
      "max_context_tokens": 1000000,
      "batch_size": 1
    },
    "teacher": {
      "family": "claude",
      "model": "claude-stub",
      "endpoint": "local",
      "max_context_tokens": 200000,
      "batch_size": 1
    },
    "student": {
      "family": "qwen2.5",
      "model": "Qwen2.5-Coder-7B-Instruct",
      "endpoint": "local",
      "max_context_tokens": 32768,
      "batch_size": 8
    }
  },
  "retrieval": {
    "vector_backend": "local-jsonl",
    "embedding_endpoint": "local"
  }
}
```

When `endpoint` is `"local"`, both LLM and embedding clients use deterministic
offline stubs — useful for CI, dry-runs, and testing the pipeline structure
without incurring API costs.

---

## 13. Environment variable reference

distillme does not read environment variables automatically.  All configuration
is explicit in the JSON config file.  However, the external servers and proxies
you connect to typically require environment variables:

| Variable | Used by | Description |
|---|---|---|
| `OPENAI_API_KEY` | LiteLLM, OpenAI SDK | OpenAI API authentication |
| `ANTHROPIC_API_KEY` | LiteLLM, Anthropic SDK | Anthropic Claude authentication |
| `GOOGLE_API_KEY` | LiteLLM, Google SDK | Google Gemini authentication |
| `MISTRAL_API_KEY` | LiteLLM | Mistral API authentication |
| `COHERE_API_KEY` | LiteLLM | Cohere API authentication |
| `HUGGING_FACE_HUB_TOKEN` | text-embeddings-inference | HF model access token |

Set these in your shell before starting the proxy or inference server, not in
the distillme config file (which should not contain secrets).
