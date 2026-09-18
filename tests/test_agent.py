from typing import Any, cast
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletionUserMessageParam

from zipcoil import Agent, tool


def test_agent_rejects_async_tools() -> None:
    @tool
    def sync_tool(x: int) -> int:
        return x

    @tool
    async def async_tool(x: int) -> int:
        return x

    mock_client = Mock()

    with pytest.raises(ValueError, match=r"async_tool.*?an async function"):
        Agent(model="gpt-4", client=mock_client, tools=[sync_tool, async_tool])


def test_agent_non_stream_still_returns_chat_completion() -> None:
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    mock_client = Mock()

    first = Mock()
    first.choices = [Mock()]
    first.choices[0].finish_reason = "tool_calls"
    first.choices[0].message.tool_calls = [Mock(id="call_1", function=Mock(name="add", arguments='{"a": 1, "b": 2}'))]
    first.choices[0].message.tool_calls[0].function.name = "add"
    first.choices[0].message.tool_calls[0].function.arguments = '{"a": 1, "b": 2}'

    second = Mock()
    second.choices = [Mock()]
    second.choices[0].finish_reason = "stop"
    second.choices[0].message.content = "Done"

    mock_client.chat.completions.create.side_effect = [first, second]

    agent = Agent(model="gpt-4", client=mock_client, tools=[add])
    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Please add 1 and 2"})]

    result = cast(Any, agent.run(messages=messages))

    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.content == "Done"
