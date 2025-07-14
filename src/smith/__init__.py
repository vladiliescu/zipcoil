"""Smith - Simplifies OpenAI tool usage."""

from .core import _parse_docstring_args, _type_to_json_schema, tool

__version__ = "0.1.0"

# Export the main decorator
__all__ = ["tool"]