# Zipcoil

**Zipcoil** is a Python library that simplifies OpenAI tool usage, helping developers build simple AI agents with ease. It provides a clean, decorator-based approach to define tools and an `Agent` class that handles the OpenAI tool-calling loop automatically.

## Why Zipcoil?

Building AI agents that can use tools typically involves:
- Converting Python functions to OpenAI's JSON schema format 😕
- Handling the complex tool-calling conversation flow 🙁
- Managing multiple iterations of tool calls and responses ☹️
- Dealing with error handling and edge cases 😣

Zipcoil eliminates this boilerplate by providing:
- A **simple `@tool` decorator** to help convert Python functions into OpenAI tools
- **Automatic schema generation** from type hints and docstrings
- **Built-in agent loop** that handles tool calling iterations
- **Rich type-hint support**, including Optional, Union, Enum, and more
- **Error handling** for malformed tool calls and execution errors
- **Minimal dependencies**, built on top of the official OpenAI library
- Works with both `OpenAI` and `AzureOpenAI` clients, in both `sync` and `async` modes

Zipcoil keeps its scope small: tool schemas and the tool-calling loop for OpenAI and Azure OpenAI's Chat Completions API. Your application owns conversation storage and workflow orchestration. Zipcoil doesn't implement the Responses API yet.

## Installation

Zipcoil requires Python 3.11 or higher.

```bash
pip install zipcoil
```

## Quick Start

Set `OPENAI_API_KEY` in your environment, then define a tool and give it to an agent:

```python
from openai import OpenAI
from zipcoil import Agent, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


agent = Agent(model="gpt-4o", client=OpenAI(), tools=[add])
messages = [{"role": "user", "content": "What is 17 + 25?"}]
result = agent.run(messages)
print(result.choices[0].message.content)
```

The decorator builds the tool schema from type hints and the docstring. The agent calls the function when requested by the model and sends the result back, repeating until the model finishes or the iteration limit is reached.

[Async](#async) · [Azure OpenAI](#azure-openai) · [Streaming](#streaming-output)

## Async

Use `AsyncAgent` with `AsyncOpenAI`. This complete example uses the same `OPENAI_API_KEY` environment variable:

```python
import asyncio

from openai import AsyncOpenAI
from zipcoil import AsyncAgent, tool


@tool
async def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


async def main() -> None:
    async with AsyncOpenAI() as client:
        agent = AsyncAgent(model="gpt-4o", client=client, tools=[add])
        messages = [{"role": "user", "content": "What is 17 + 25?"}]
        result = await agent.run(messages)
        print(result.choices[0].message.content)


asyncio.run(main())
```

`AsyncAgent` accepts both sync and async tools and awaits async tools. Sync tools run inline, so blocking I/O in them will block the event loop.

## Azure OpenAI

Set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, and `AZURE_OPENAI_DEPLOYMENT` in your environment. Using `add` and `messages` from the synchronous [quickstart](#quick-start):

```python
import os

from openai import AzureOpenAI
from zipcoil import Agent

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
agent = Agent(model=os.environ["AZURE_OPENAI_DEPLOYMENT"], client=client, tools=[add])
result = agent.run(messages)
print(result.choices[0].message.content)
```

Pass your **deployment name** as `model`; it can differ from the underlying model name. Zipcoil leaves authentication, endpoints, and API-version configuration to the SDK client.

For async Azure usage, pass an `AsyncAzureOpenAI` client to `AsyncAgent` with the same configuration.

## Advanced Usage

### Strict Mode

`@tool` uses `strict=True` by default, following [OpenAI's recommendation](https://developers.openai.com/api/docs/guides/function-calling#strict-mode).
OpenAI recommends strict mode but does not require it for Chat Completions tool calls.

Strict mode constrains the model's arguments to the generated schema. For example,
an `int` parameter must receive a JSON integer. With `strict=False`, OpenAI still
receives the schema and tries to follow it, but may supply a wrong type or omit an
argument. Zipcoil checks input annotations for strict-mode compatibility when
the function is decorated. It does not validate the values passed when a tool runs.

#### Dictionaries

A function accepting `dict[str, int]` might receive `{"apples": 2, "oranges": 3}`
on one call and `{"books": 4}` on another. Its keys are not known when the function
is decorated.

OpenAI's strict mode requires object schemas to declare every permitted key and
forbid additional keys. Forbidding additional keys without declaring any would
allow only an empty dictionary, which would not describe this function correctly.

Use non-strict mode for dictionary inputs:

```python
@tool(strict=False)
def total(values: dict[str, int]) -> int:
    """Add the supplied values."""
    return sum(values.values())
```

In this example, Zipcoil exposes `values` as an object accepting any string key, with integer
values. Dictionary values can also have supported list, union, or dictionary
annotations; Zipcoil exposes them recursively. JSON object keys are always
strings, so use string keys in your dictionary annotations. Zipcoil does not
convert keys into other Python types.

#### Lists

`list[int]` describes each element as an integer and works in strict mode. A bare
`list` specifies no element type: it might contain `[2, "hello", None]`, or nested
lists and dictionaries. Its schema allows any JSON value as an element, which
requires non-strict mode:

```python
@tool(strict=False)
def count(values: list) -> int:
    """Count the supplied values."""
    return len(values)
```

If you know the possible element types, use an annotation such as
`list[int | str | None]` and keep strict mode.

#### Any

`Any` allows any JSON value: a number, string, boolean, null, list, or object.
Zipcoil describes it with an empty schema, `{}`, which places no restrictions on
the value. This requires non-strict mode:

```python
from typing import Any

@tool(strict=False)
def echo(value: Any) -> Any:
    """Return the supplied value unchanged."""
    return value
```

Set `strict=False` on a tool when any input is `Any`, a dictionary, or a bare list,
including inside another list or union. For example, `list[dict[str, int]]` and
`list[Any]` also need it. With `strict=True`, Zipcoil raises `ValueError` when the
function is decorated, naming the tool and parameter and recommending
`@tool(strict=False)`. This happens before any API request.

The setting applies to **all arguments of that tool**. Other tools select their
own strict setting independently.

These restrictions concern inputs from the model. A tool can return a dictionary
or list, or have an `Any` return annotation, regardless of its strict setting.

### Complex Type Support

Zipcoil supports various Python types including enums, optionals, and unions:

```python
from enum import Enum
from typing import Optional, List, Dict

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@tool(strict=False)
def create_task(
    title: str,
    description: Optional[str],
    priority: Priority,
    tags: List[str],
    metadata: Dict[str, str]
) -> str:
    """Create a new task.

    Args:
        title: Task title
        description: Optional task description
        priority: Task priority level
        tags: List of tags for the task
        metadata: Additional metadata as key-value pairs
    """
    return f"Created task '{title}' with priority {Priority(priority).name}"
```

This example uses `strict=False` because `metadata` is a dictionary. Enum arguments
arrive as their JSON values, so `Priority(priority)` converts the value to an enum.

### Error Handling

Zipcoil automatically handles tool execution errors:

```python
@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a: Numerator
        b: Denominator
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# The agent will catch the error and include it in the conversation
```

### Custom Agent Configuration

You can pass additional parameters to the underlying OpenAI API:

```python
result = agent.run(
    messages=messages,
    temperature=0.7,
    max_completion_tokens=1000,
    max_iterations=5  # Limit tool calling iterations
)
```

### Streaming Output

Set `stream=True` to get a chunk stream compatible with `chat.completions.create(stream=True)`.
Zipcoil still handles tool calls between streamed model turns. Using the synchronous `agent` and `messages` from [Quick Start](#quick-start):

```python
stream = agent.run(messages=messages, stream=True)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

For `AsyncAgent`, `await agent.run(..., stream=True)` and then `async for chunk in stream: ...`.


## Type Support

Zipcoil automatically converts Python types to OpenAI's JSON schema:

| Python Type | JSON Schema Type | Notes |
|-------------|------------------|--------|
| `str` | `string` | |
| `int` | `integer` | |
| `float` | `number` | |
| `bool` | `boolean` | |
| `list[T]` | `array` with `items` | Describes element type recursively |
| `list` | `array` with unrestricted `items` | Requires `strict=False` |
| `dict[str, T]` | `object` with typed `additionalProperties` | Requires `strict=False`; describes values recursively |
| `dict` | `object` with unrestricted `additionalProperties` | Requires `strict=False` |
| `Any` | Empty schema (`{}`), allowing any JSON value | Requires `strict=False` |
| `Optional[T]`, `T \| None` | `anyOf` for T and `null` | Also allows null for nullable enums |
| `Union[T, U]`, `T \| U` | `anyOf` | Describes each alternative recursively |
| `Enum` | Primitive type with `enum` | Extracts enum values |

## API Reference

### `@tool` Decorator

Use `@tool`, `@tool()`, or `@tool(strict=True)` for strict mode. Use
`@tool(strict=False)` for non-strict mode. Both forms support synchronous and
asynchronous functions. See [Strict Mode](#strict-mode) for the tradeoffs and
dictionary/list examples.

Converts a Python function into an OpenAI tool. The function should:
- Have type hints for all parameters
- Have a docstring with Google-style Args section

Example:

```python
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"The weather in {city} is 22°{unit[0].upper()}"
```

### `Agent` and `AsyncAgent` Classes

```python
Agent(
    model: Uniont[str, ChatModel],
    client: OpenAI,
    tools: Iterable[ToolProtocol]
)

AsyncAgent(
    model: Union[str, ChatModel],
    client: AsyncOpenAI,
    tools: Iterable[Union[ToolProtocol, AsyncToolProtocol]]
)
```

The main abstraction of the agentic event loop. It will take in a model name (more on this below), an OpenAI or AzureOpenAI client, and a list of tools decorated with the `@tool` decorator.

While `Agent` will accept only sync tools and will reject async ones, `AsyncAgent` will accept both sync and async tools and will `await` async tools. Use `@tool` for both sync and async functions.

Note: As opposed to standard OpenAI usage, Zipcoil associates the model with an agent to avoid having to specify it every time you call `run`.

#### `Agent.run()`

Runs the agentic loop, calling all tools as needed and iterating until the underlying model doesn't need to call any tools anymore, until it's ready to return a ChatCompletion.

**Parameters:**
- `max_iterations`: Maximum number of tool calling iterations (default: 10)
- `stream`: When `True`, `run()` returns streamed `ChatCompletionChunk` objects (default: `False`)
- All other parameters are passed through to OpenAI's chat completion API
- Return type:
  - `stream=False`: OpenAI [ChatCompletion](https://platform.openai.com/docs/api-reference/chat/object)
  - `stream=True`: chunk stream compatible with `chat.completions.create(stream=True)`

## Error Handling

Zipcoil handles several types of errors gracefully:

1. **Tool execution errors**: Caught and passed back to the model as error messages
2. **JSON parsing errors**: Invalid tool arguments are reported to the model
3. **Missing tools**: Requests for non-existent tools return error messages
4. **Iteration limits**: Prevents infinite loops with configurable max iterations

## Contributing

Development setup:

```bash
# Install development dependencies
uv sync --group dev

# Run tests
uv run pytest

# Format code
uv run ruff format src/ tests/

# Run checks
just check
```

## Requirements

- Python 3.11+
- OpenAI Python library (≥1.95.1)
- docstring-parser (≥0.16)

## License

Copyright (c) 2025-2026 Vlad Iliescu.

This project is open-source, licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

---
