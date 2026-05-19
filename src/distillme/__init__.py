"""Repository-specialized knowledge distillation pipeline."""

from distillme.config import PipelineConfig
from distillme.inference import LLMClient, make_client
from distillme.orchestration import DistillationPipeline

__all__ = ["DistillationPipeline", "LLMClient", "PipelineConfig", "make_client"]
