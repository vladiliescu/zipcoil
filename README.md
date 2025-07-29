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
- **Type safety** with comprehensive type hints including Optional, Union, Enum, and more
- **Error handling** for malformed tool calls and execution errors
- A very bearable **lightness of being**, using minimal dependencies, built on top of the official OpenAI library
- Works with both `OpenAI` and `AzureOpenAI` clients

## Installation

Zipcoil requires Python 3.11 or higher.

```bash
pip install zipcoil
```

## Quick Start

Here's a simple example of creating an AI agent with tools:

```python
import os
from enum import Enum

from openai import AzureOpenAI, OpenAI

from zipcoil import Agent, tool

# Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# or the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


# Define tools using the @tool decorator
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Your weather API call here
    return f"The weather in {city} is 22°{unit[0].upper()}"


class MathOp(Enum):
    ADD = 1
    SUBTRACT = 2
    MULTIPLY = 3
    DIVIDE = 4


@tool
def calculate(x: float, y: float, operation: MathOp) -> float:
    """Perform a mathematical calculation.

    Args:
        x: First number
        y: Second number
        operation: Operation to perform (add, subtract, multiply, divide)
    """
    # normalise int -> MathOp
    if isinstance(operation, int):
        try:
            operation = MathOp(operation)
        except ValueError as exc:
            raise ValueError(f"Unsupported operation value: {operation}") from exc

    operations = {
        MathOp.ADD: x + y,
        MathOp.SUBTRACT: x - y,
        MathOp.MULTIPLY: x * y,
        MathOp.DIVIDE: x / y if y != 0 else float("inf"),
    }
    return operations.get(operation, 0)


# Create an agent with tools
agent = Agent(model="gpt-4o", client=client, tools=[get_weather, calculate])

# Run a conversation
messages = [{"role": "user", "content": "What's the weather in Paris? Also calculate 15 * 23."}]

result = agent.run(messages)
print(result.choices[0].message.content)

```

## Advanced Usage

### Complex Type Support

Zipcoil supports various Python types including enums, optionals, and unions:

```python
from enum import Enum
from typing import Optional, List, Dict

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@tool
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
    return f"Created task '{title}' with priority {priority.name}"
```

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


## Type Support

Zipcoil automatically converts Python types to OpenAI's JSON schema:

| Python Type | JSON Schema Type | Notes |
|-------------|------------------|--------|
| `str` | `string` | |
| `int` | `integer` | |
| `float` | `number` | |
| `bool` | `boolean` | |
| `list` | `array` | |
| `dict` | `object` | |
| `Optional[T]` | `[T, "null"]` | Union with null |
| `Union[T, U]` | Mixed type | For Optional types |
| `Enum` | `enum` | Extracts enum values |

## API Reference

### `@tool` Decorator

Converts a Python function into an OpenAI tool. The function must:
- Have type hints for all parameters
- Have a docstring with Google-style Args section
- Be synchronous (async functions are not yet supported)

e.g.

```python
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Your weather API call here
    return f"The weather in {city} is 22°{unit[0].upper()}"
```

### `Agent` Class

```python
Agent(
    model: str | ChatModel,
    client: OpenAI,
    tools: Iterable[ToolProtocol]
)
```

The main abstraction of the agentic event loop. It will take in a model name (more on this below), an OpenAI or AzureOpenAI client, and a list of tools decorated with the `@tool` decorator.

Note: As opposed to standard OpenAI usage, Zipcoil associates the model with an agent to avoid having to specify it every time you call `run`.

#### `Agent.run()`

Runs the agentic loop, calling all tools as needed and iterating until the underlying model doesn't need to call any tools anymore, until it's ready to return a ChatCompletion.

**Parameters:**
- `max_iterations`: Maximum number of tool calling iterations (default: 10)
- All other parameters are passed through to OpenAI's chat completion API
- Returns the standard OpenAI [ChatCompletion](https://platform.openai.com/docs/api-reference/chat/object) object

## Error Handling

Zipcoil handles several types of errors gracefully:

1. **Tool execution errors**: Caught and passed back to the model as error messages
2. **JSON parsing errors**: Invalid tool arguments are reported to the model
3. **Missing tools**: Requests for non-existent tools return error messages
4. **Iteration limits**: Prevents infinite loops with configurable max iterations

## Contributing

Contributions are welcome! Please see our development setup:

```bash
# Install development dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run ruff format src/ tests/
```

## Requirements

- Python 3.11+
- OpenAI Python library (≥1.0.0)
- docstring-parser (≥0.16)

## License

This project is open-source, licensed under the GNU Lesser General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

**Zipcoil** - Making AI tool usage as simple as decorating a function.