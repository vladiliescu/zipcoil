import asyncio
import functools
import inspect
import types
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from docstring_parser import DocstringStyle, ParseError, parse
from openai.types.chat import ChatCompletionToolParam

from zipcoil.types import AsyncToolProtocol, ToolDecoratorProtocol, ToolProtocol


def _enum_type_to_json_schema(type_hint):
    """Convert Enum types to JSON schema format."""

    # If the enum has mixed types, default to string.
    member_types = {type(member.value) for member in type_hint}
    if len(member_types) > 1:
        return {"type": "string", "enum": [str(member.value) for member in type_hint]}

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


def _type_to_json_schema(type_hint: Any) -> dict[str, Any]:
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
    elif type_hint is type(None):
        return {"type": "null"}
    elif type_hint == list or get_origin(type_hint) is list:
        args = get_args(type_hint)
        return {"type": "array", "items": _type_to_json_schema(args[0]) if args else {}}
    elif type_hint == dict or get_origin(type_hint) is dict:
        args = get_args(type_hint)
        return {"type": "object", "additionalProperties": _type_to_json_schema(args[1]) if args else {}}
    elif get_origin(type_hint) in (Union, types.UnionType):
        return {"anyOf": [_type_to_json_schema(arg) for arg in get_args(type_hint)]}
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


@overload
def tool(func: Callable[..., Awaitable[Any]], *, strict: bool = True) -> AsyncToolProtocol: ...


@overload
def tool(func: Callable[..., Any], *, strict: bool = True) -> ToolProtocol: ...


@overload
def tool(func: None = None, *, strict: bool = True) -> ToolDecoratorProtocol: ...


def tool(
    func: Callable[..., Any] | None = None, *, strict: bool = True
) -> ToolProtocol | AsyncToolProtocol | ToolDecoratorProtocol:
    """
    Decorator that extracts function metadata and converts it to the OpenAI function-calling JSON schema format.

    Strict mode defaults to True. Use @tool(strict=False) for dictionary or bare-list inputs.
    """
    if func is None:
        return functools.partial(tool, strict=strict)

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper = async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper = sync_wrapper

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    docstring = inspect.getdoc(func) or ""
    description = docstring.split("\n\n")[0].strip() if docstring else ""
    arg_descriptions = _parse_docstring_args(docstring)

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in type_hints:
            type_hint = type_hints[param_name]
            json_type = _type_to_json_schema(type_hint)

            properties[param_name] = json_type
            properties[param_name]["description"] = arg_descriptions.get(param_name, "")
            # mark all parameters as required to comply with strict=True
            required.append(param_name)

    tool_schema: ChatCompletionToolParam = {
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
            "strict": strict,
        },
    }

    wrapper.tool_schema = tool_schema  # type: ignore[attr-defined]

    # Help static type checkers
    if asyncio.iscoroutinefunction(func):
        return cast(AsyncToolProtocol, wrapper)
    return cast(ToolProtocol, wrapper)
