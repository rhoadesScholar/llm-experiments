"""llm_experiments: A repository of experiments inspecting LLM's, particularly focused on detecting signs of human-like self-awareness via psychodynamic-inspired prompting.

A Python package for a repository of experiments inspecting llm's, particularly focused on detecting signs of human-like self-awareness via psychodynamic-inspired prompting..
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-experiments")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Jeff Rhoades"
__email__ = "rhoadesj@alumni.harvard.edu"

from .core import LlmExperiments

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "LlmExperiments",
]
