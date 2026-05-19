"""Repository-specialized knowledge distillation pipeline."""

from distillme.config import PipelineConfig
from distillme.orchestration import DistillationPipeline

__all__ = ["DistillationPipeline", "PipelineConfig"]
