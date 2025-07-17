import functools
import inspect
import json
import types
from enum import Enum
from typing import Dict, Iterable, List, Literal, Optional, Union, get_args, get_origin, get_type_hints

import httpx
from docstring_parser import DocstringStyle, ParseError, parse
from openai import NOT_GIVEN, NotGiven, OpenAI
from openai._types import Body, Headers, Query
from openai.types import ChatModel, Metadata, ReasoningEffort
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    completion_create_params,
)


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


class Agent:
    """An abstractization of the OpenAI tool-calling loop"""

    def __init__(
        self,
        model: Union[str, ChatModel],
        client: OpenAI,
        tools: list,
    ) -> None:
        self.model = model
        self.client = client
        self.tools = tools

    def _call_function(self, name: str, args: dict):
        for potential_tool in self.tools:
            if potential_tool._tool_schema["function"]["name"] == name:
                try:
                    return potential_tool(**args)
                except Exception as e:
                    return f"Error executing tool `{name}`: {str(e)}"

        return f"Tool `{name}` not found"

    def run(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        audio: Optional[ChatCompletionAudioParam] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
        modalities: Optional[List[Literal["text", "audio"]]] | NotGiven = NOT_GIVEN,
        # n: Optional[int] | NotGiven = NOT_GIVEN,
        # parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        prediction: Optional[ChatCompletionPredictionContentParam] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        reasoning_effort: Optional[ReasoningEffort] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        # stream: Optional[Literal[False]] | Literal[True] | NotGiven = NOT_GIVEN,
        # stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        # tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        # tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion:
        # TODO: add a limit to the number of messages
        mutable_messages = list(messages)
        while True:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=mutable_messages,
                tools=[potential_tool._tool_schema for potential_tool in self.tools],
                n=1,  # Only one completion at a time otherwise the logic gets messy
                audio=audio,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                metadata=metadata,
                modalities=modalities,
                prediction=prediction,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
                seed=seed,
                service_tier=service_tier,
                stop=stop,
                store=store,
                temperature=temperature,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )

            if completion.choices[0].finish_reason == "stop":
                return completion
            elif completion.choices[0].finish_reason == "tool_calls":
                mutable_messages.append(completion.choices[0].message)
                for tool_call in completion.choices[0].message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    result = self._call_function(name, args)

                    mutable_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
            else:
                raise ValueError(f"Unexpected finish reason: {completion.choices[0].finish_reason}")
