import functools
import inspect
import re
from typing import get_args, get_origin, get_type_hints


def _type_to_json_schema(type_hint):
    """Convert Python type hints to JSON schema types."""
    from typing import Union

    if type_hint == str:
        return "string"
    elif type_hint == int:
        return "integer"
    elif type_hint == float:
        return "number"
    elif type_hint == bool:
        return "boolean"
    elif type_hint == list:
        return "array"
    elif type_hint == dict:
        return "object"
    elif get_origin(type_hint) is list:
        return "array"
    elif get_origin(type_hint) is dict:
        return "object"
    elif get_origin(type_hint) is Union:
        # Handle Optional[T] which is Union[T, None]
        args = get_args(type_hint)
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], return the schema for T
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return _type_to_json_schema(non_none_type)
    # Default to string for unknown types
    return "string"

def _parse_docstring_args(docstring) -> dict:
    """Parse the Args section from a function's docstring.

    Returns:
        A dictionary mapping argument names to their descriptions.
    """
    if not docstring:
        return {}

    # Look for Args: section
    args_section_match = re.search(r'Args:\s*\n(.*?)(?:\n\n|\n[A-Z][a-z]+:|\Z)', docstring, re.DOTALL)
    if not args_section_match:
        return {}

    args_text = args_section_match.group(1)
    arg_descriptions = {}

    # Parse each argument line (format: "arg_name: description")
    for line in args_text.split('\n'):
        line = line.strip()
        if ':' in line:
            arg_name, description = line.split(':', 1)
            arg_descriptions[arg_name.strip()] = description.strip()

    return arg_descriptions

def tool(func):
    """
    Decorator that extracts function metadata and converts it to OpenAI function calling JSON schema format.
    """
    # Import Union here to avoid circular import issues
    from typing import Union

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Get function signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Parse docstring for descriptions
    docstring = inspect.getdoc(func) or ""
    description = docstring.split('\n\n')[0].strip() if docstring else func.__name__
    arg_descriptions = _parse_docstring_args(docstring)

    # Build parameters schema
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in type_hints:
            type_hint = type_hints[param_name]
            json_type = _type_to_json_schema(type_hint)

            properties[param_name] = {
                "type": json_type,
                "description": arg_descriptions.get(param_name, param_name)
            }

            # Check if parameter is required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

    # Build the complete tool schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # Store the schema on the function for easy access
    wrapper._tool_schema = tool_schema

    return wrapper
