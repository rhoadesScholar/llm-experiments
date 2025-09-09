"""Introspection by Telephone experiment module.

This module implements the complete Introspection by Telephone experiment
for investigating latent self-curiosity and self-awareness in large language models.
"""

from .core import Context, Conversation
from .contexts import (
    get_all_contexts, 
    get_context_by_name, 
    get_embodied_contexts, 
    get_ai_assistant_contexts,
    INTROSPECTION_QUESTION,
    CONTEXTS
)
from .llm_interface import LLMInterface, EvaluatorLLM
from .distillation import DistillationPipeline, DistillationComparison
from .experiment import IntrospectionExperiment

__all__ = [
    "Context",
    "Conversation", 
    "get_all_contexts",
    "get_context_by_name",
    "get_embodied_contexts",
    "get_ai_assistant_contexts", 
    "INTROSPECTION_QUESTION",
    "CONTEXTS",
    "LLMInterface",
    "EvaluatorLLM",
    "DistillationPipeline",
    "DistillationComparison",
    "IntrospectionExperiment"
]
