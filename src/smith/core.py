import functools
import inspect
from typing import Union, get_args, get_origin, get_type_hints

from docstring_parser import DocstringStyle, ParseError, parse


def _type_to_json_schema(type_hint):
    """Convert Python type hints to JSON schema types."""
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

    try:
        parsed = parse(docstring, DocstringStyle.GOOGLE)
    except ParseError:
        return {}
    return {param.arg_name: param.description for param in parsed.params}


def tool(func):
    """
    Decorator that extracts function metadata and converts it to OpenAI function calling JSON schema format.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    docstring = inspect.getdoc(func) or ""
    description = docstring.split("\n\n")[0].strip() if docstring else func.__name__
    arg_descriptions = _parse_docstring_args(docstring)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in type_hints:
            type_hint = type_hints[param_name]
            json_type = _type_to_json_schema(type_hint)

            properties[param_name] = {"type": json_type, "description": arg_descriptions.get(param_name, param_name)}

            # Check if parameter is required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

    tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    wrapper._tool_schema = tool_schema

    return wrapper
