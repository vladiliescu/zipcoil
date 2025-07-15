import functools
import inspect
import types
from enum import Enum
from typing import Union, get_args, get_origin, get_type_hints

from docstring_parser import DocstringStyle, ParseError, parse


def _enum_type_to_json_schema(type_hint):
    """Convert Enum types to JSON schema format."""

    # Use the first member to decide the underlying primitive type.
    sample_value = next(iter(type_hint)).value
    if isinstance(sample_value, str):
        json_type = "string"
    elif isinstance(sample_value, bool):
        json_type = "boolean"
    elif isinstance(sample_value, int):
        json_type = "integer"
    elif isinstance(sample_value, float):
        json_type = "number"
    else:
        json_type = "string"  # fallback

    return {"type": json_type, "enum": [member.value for member in type_hint]}


def _type_to_json_schema(type_hint) -> dict:
    """Convert Python type hints to JSON schema types."""
    if inspect.isclass(type_hint) and issubclass(type_hint, Enum):
        return _enum_type_to_json_schema(type_hint)

    if type_hint == str:
        return {"type": "string"}
    elif type_hint == int:
        return {"type": "integer"}
    elif type_hint == float:
        return {"type": "number"}
    elif type_hint == bool:
        return {"type": "boolean"}
    elif type_hint == list or get_origin(type_hint) is list:
        return {"type": "array"}
    elif type_hint == dict or get_origin(type_hint) is dict:
        return {"type": "object"}
    elif get_origin(type_hint) is Union or isinstance(type_hint, types.UnionType):
        # Handle Optional[T] which is Union[T, None] or T | None
        args = get_args(type_hint)
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], return the schema for T
            non_none_type = args[0] if args[1] is type(None) else args[1]
            schema = _type_to_json_schema(non_none_type)
            schema["type"] = [schema["type"], "null"] if isinstance(schema, dict) else [schema, "null"]
            return schema
    # Default to string for unknown types
    return {"type": "string"}


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
    description = docstring.split("\n\n")[0].strip() if docstring else ""
    arg_descriptions = _parse_docstring_args(docstring)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in type_hints:
            type_hint = type_hints[param_name]
            json_type = _type_to_json_schema(type_hint)

            properties[param_name] = json_type
            properties[param_name]["description"] = arg_descriptions.get(param_name, "")
            # mark all parameters as required to comply with strict=True
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
