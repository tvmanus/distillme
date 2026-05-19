# distillme

Experimental code distillation pipeline for building a repository-specialized software engineering advisor.

## What this provides

`distillme` is a Python orchestration framework for a three-stage heterogeneous-LLM knowledge distillation pipeline:

1. **Investigator**: indexes a Java-oriented repository and emits evidence-scoped archaeology documents.
2. **Teacher**: turns investigator findings and retrieved source context into grounded synthetic instruction data.
3. **Student**: prepares reproducible training and evaluation artifacts for a compact 7B coding advisor model.

The implementation focuses on factual grounding, explicit uncertainty, retrieval-backed generation, resumability, observability, and validation before training.

## Current capabilities

- Multi-layer local indexes for artifacts, chunks, symbol relationships, call hints, dependency/config/docs/tests, and error-flow evidence.
- Hybrid retrieval using deterministic dense-hash, sparse token, and symbol signals.
- Mandatory investigator documents such as `architecture_overview.md`, `domain_model.md`, `security_analysis.md`, and `exception_taxonomy.md`.
- Synthetic instruction dataset records using the requested schema fields.
- Dataset validation for source grounding, duplicate questions, retrieved context consistency, and uncertainty calibration.
- Training and benchmark plan artifacts for QLoRA/SFT/curriculum/retrieval-aware specialization.
- JSONL trace logging and resumable stage state.

## Quick start

```bash
python -m distillme.cli init \
  --repository /absolute/path/to/java/repository \
  --workdir /absolute/path/to/distillme-workdir \
  --output /absolute/path/to/config.json

python -m distillme.cli run --config /absolute/path/to/config.json
```

A sample configuration is available at `configs/local.distillme.json`.

## Output layout

The configured workdir contains:

- `index/`: artifact, chunk, graph, and manifest JSONL/JSON files
- `investigator/`: required evidence-backed markdown findings
- `dataset/`: `instruction_dataset.jsonl`, manifest, and validation report
- `training/`: student fine-tuning plan
- `evaluation/`: benchmark definitions
- `logs/`: trace events
- `state.json`: resumable stage state

## Model heterogeneity

The configuration validates distinct model families for investigator, teacher, and student roles. Defaults use Gemini-family investigator, Claude-family teacher, and Qwen2.5 7B student placeholders; endpoints can point at local, distributed, or managed inference services.
