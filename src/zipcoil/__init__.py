"""Zipcoil - Simplifies OpenAI tool usage."""

from .agent import Agent
from .async_agent import AsyncAgent
from .core import async_tool, tool

__version__ = "0.1.0"

# Export the main decorator
__all__ = ["tool", "async_tool", "Agent", "AsyncAgent"]
