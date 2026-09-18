"""Live checks of the current tool schemas against Azure OpenAI.

Run explicitly (this file is excluded from normal pytest discovery):
    uv run --no-sync --env-file tests/code_samples/.env pytest -v -s tests/code_samples/schema_baseline.py

Each case runs Agent and checks the value and type received by its tool.
At most two API requests are made per case: a tool call and a final response.
Failures reveal current limitations; they are not marked as expected failures.
Dictionary and bare-list cases explicitly disable strict mode.
"""

import json
import os
from collections.abc import AsyncIterator, Iterator
from enum import Enum
from typing import Any, Literal

import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from zipcoil import Agent, AsyncAgent, tool


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
    ("annotation", "expected", "strict"),
    [
        pytest.param(str, "hello", True, id="string-control"),
        pytest.param(list[int], [2, 3, 5], True, id="typed-list"),
        pytest.param(list[list[int]], [[2, 3], [5]], True, id="nested-list"),
        pytest.param(int | float, 7, True, id="union-integer"),
        pytest.param(int | float, 2.5, True, id="union-float"),
        pytest.param(list[int | float], [2, 3.5], True, id="list-of-unions"),
        pytest.param(int | float | None, None, True, id="nullable-numeric-union"),
        pytest.param(Color | None, "red", True, id="nullable-enum-value"),
        pytest.param(Color | None, None, True, id="nullable-enum-null"),
        pytest.param(dict[str, int], {"apples": 2, "oranges": 3}, False, id="dictionary"),
        pytest.param(dict, {"apples": 2, "labels": ["red", None]}, False, id="bare-dictionary"),
        pytest.param(dict[str, list[int | float]], {"values": [2, 3.5]}, False, id="dictionary-of-lists"),
        pytest.param(list[dict[str, int]], [{"apples": 2}, {"oranges": 3}], False, id="list-of-dictionaries"),
        pytest.param(dict[str, int] | None, None, False, id="nullable-dictionary"),
        pytest.param(list, [2, "hello", None, [3], {"apples": 2}], False, id="bare-list"),
        pytest.param(Any, 7, True, id="any-number"),
        pytest.param(Any, {"apples": 2}, True, id="any-object"),
        pytest.param(Literal["red", "blue"], "red", True, id="literal"),
    ],
)
def test_tool_argument(client: AzureOpenAI, annotation: Any, expected: Any, strict: bool) -> None:
    received: list[Any] = []

    @tool(strict=strict)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_async_non_strict_tool(stream: bool) -> None:
    expected = {"values": [2, "hello", None, {"apples": 3}]}
    received: list[dict[str, list]] = []

    @tool(strict=False)
    async def echo(value: dict[str, list]) -> str:
        """Return the supplied value unchanged."""
        received.append(value)
        return json.dumps(value)

    async with AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        timeout=30,
        max_retries=0,
    ) as client:
        agent = AsyncAgent(model="gpt-4o", client=client, tools=[echo])
        result = await agent.run(
            messages=[
                {
                    "role": "user",
                    "content": f"Call echo exactly once with value {json.dumps(expected)}. "
                    "Preserve JSON types. After the tool responds, say Done without calling it again.",
                }
            ],
            stream=stream,
            temperature=0,
            max_iterations=2,
        )
        if stream:
            assert isinstance(result, AsyncIterator)
            async for chunk in result:
                assert isinstance(chunk, ChatCompletionChunk)
        else:
            assert isinstance(result, ChatCompletion)

    assert json.dumps(received, sort_keys=True) == json.dumps([expected], sort_keys=True)
