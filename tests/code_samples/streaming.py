import logging
import os
from enum import Enum

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from zipcoil import Agent, tool

logging.basicConfig(
    level=logging.INFO,  # show INFO and above
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# or the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY") or "",
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE") or "",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "",
)


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city
        unit: Temperature unit (celsius or fahrenheit)
    """
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


agent = Agent(model="gpt-4o", client=client, tools=[get_weather, calculate])

messages: list[ChatCompletionUserMessageParam] = [
    {"role": "user", "content": "What's the weather in Paris? Also calculate 15 * 23."}
]


def on_text_delta(delta: str) -> None:
    print(delta, end="", flush=True)


result = agent.run(messages=messages, stream=True, on_text_delta=on_text_delta)

print("\n")
print(result.choices[0].message.content)
