"""Live checks of the current tool schemas against Azure OpenAI.

Run explicitly (this file is excluded from normal pytest discovery):
    uv run --no-sync --env-file tests/code_samples/.env pytest -v -s tests/code_samples/schema_baseline.py

Each case runs Agent and checks the value and type received by its tool.
At most two API requests are made per case: a tool call and a final response.
Failures reveal current limitations; they are not marked as expected failures.
"""

import json
import os
from collections.abc import Iterator
from enum import Enum
from typing import Any, Literal

import pytest
from openai import AzureOpenAI

from zipcoil import Agent, tool


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@pytest.fixture(scope="module")
def client() -> Iterator[AzureOpenAI]:
    with AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        timeout=30,
        max_retries=0,
    ) as azure:
        yield azure


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(str, "hello", id="string-control"),
        pytest.param(list[int], [2, 3, 5], id="typed-list"),
        pytest.param(list[list[int]], [[2, 3], [5]], id="nested-list"),
        pytest.param(int | float, 7, id="union-integer"),
        pytest.param(int | float, 2.5, id="union-float"),
        pytest.param(list[int | float], [2, 3.5], id="list-of-unions"),
        pytest.param(int | float | None, None, id="nullable-numeric-union"),
        pytest.param(Color | None, "red", id="nullable-enum-value"),
        pytest.param(Color | None, None, id="nullable-enum-null"),
        pytest.param(dict[str, int], {"apples": 2, "oranges": 3}, id="dictionary"),
        pytest.param(list, [2, "hello", None], id="bare-list"),
        pytest.param(Any, 7, id="any-number"),
        pytest.param(Any, {"apples": 2}, id="any-object"),
        pytest.param(Literal["red", "blue"], "red", id="literal"),
    ],
)
def test_tool_argument(client: AzureOpenAI, annotation: Any, expected: Any) -> None:
    received: list[Any] = []

    @tool
    def echo(value: annotation) -> str:
        """Return the supplied value unchanged."""
        received.append(value)
        return json.dumps(value)

    print("Schema:", json.dumps(echo.tool_schema["function"]["parameters"]))
    agent = Agent(model="gpt-4o", client=client, tools=[echo])
    agent.run(
        messages=[
            {
                "role": "user",
                "content": f"Call echo exactly once with value {json.dumps(expected)}. "
                "Preserve JSON types. After the tool responds, say Done without calling it again.",
            }
        ],
        temperature=0,
        max_iterations=2,
    )
    assert len(received) == 1, f"Expected one tool execution, received: {received!r}"
    actual = received[0]
    print(f"Expected: {expected!r}; received: {actual!r}")
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
