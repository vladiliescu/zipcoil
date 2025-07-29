"""Zipcoil - Simplifies OpenAI tool usage."""

from .core import Agent, tool

__version__ = "0.1.0"

# Export the main decorator
__all__ = ["tool", "Agent"]
