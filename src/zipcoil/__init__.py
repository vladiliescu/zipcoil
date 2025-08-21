"""Zipcoil - Simplifies OpenAI tool usage."""

from importlib.metadata import version

from .agent import Agent, AsyncAgent
from .core import tool

__version__ = version("zipcoil")

# Export the main decorator
__all__ = ["tool", "Agent", "AsyncAgent"]
