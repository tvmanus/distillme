FROM python:3.11-slim

LABEL org.opencontainers.image.title="distillme" \
      org.opencontainers.image.description="Agentic repository knowledge distillation pipeline" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

WORKDIR /app

# Install build tools, then remove them to keep the image slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# Default entrypoint exposes the distillme CLI.
# Mount your repository and workdir as volumes and pass a config:
#
#   docker run --rm \
#     -v /path/to/repo:/repo:ro \
#     -v /path/to/workdir:/workdir \
#     -v /path/to/config.json:/config.json:ro \
#     distillme run --config /config.json
#
ENTRYPOINT ["python", "-m", "distillme.cli"]
CMD ["--help"]
