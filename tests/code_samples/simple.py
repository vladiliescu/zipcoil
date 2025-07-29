import logging
import os
from enum import Enum

from openai import AzureOpenAI, OpenAI

from zipcoil import Agent, tool

logging.basicConfig(
    level=logging.INFO,  # show INFO and above
    format="%(asctime)s %(levelname)s: %(message)s",
)

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
