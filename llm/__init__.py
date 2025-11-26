"""
High-level entry points for the LLM pipeline module.

The public API exposes :func:`invoke_llm`, which reads the local configuration
and automatically falls back across providers/models/api keys as defined in
``config.llm.json``.  See :mod:`pipeline.llm.client` for the implementation.
"""

from .client import LLMClient, LLMConfigError, LLMInvocationError, invoke_llm

__all__ = [
    "invoke_llm",
    "LLMClient",
    "LLMConfigError",
    "LLMInvocationError",
]
