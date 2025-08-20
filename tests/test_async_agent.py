import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from zipcoil import AsyncAgent, tool


@pytest.mark.asyncio
async def test_async_agent_with_async_tools():
    @tool
    async def add_async(a: int, b: int) -> int:
        """Add two numbers asynchronously."""
        await asyncio.sleep(0.001)  # Simulate async work
        return a + b

    @tool
    async def multiply_async(x: int, y: int) -> int:
        """Multiply two numbers asynchronously."""
        await asyncio.sleep(0.001)  # Simulate async work
        return x * y

    mock_client = AsyncMock()

    # Second call returns stop
    def side_effect(*args, **kwargs):
        if mock_client.chat.completions.create.call_count == 1:
            # First call - tool calls
            result = AsyncMock()
            result.choices = [AsyncMock()]
            result.choices[0].finish_reason = "tool_calls"
            result.choices[0].message.tool_calls = [
                AsyncMock(id="call_1", function=AsyncMock(name="add_async", arguments='{"a": 5, "b": 3}')),
                AsyncMock(id="call_2", function=AsyncMock(name="multiply_async", arguments='{"x": 4, "y": 2}')),
            ]
            result.choices[0].message.tool_calls[0].function.name = "add_async"
            result.choices[0].message.tool_calls[0].function.arguments = '{"a": 5, "b": 3}'
            result.choices[0].message.tool_calls[1].function.name = "multiply_async"
            result.choices[0].message.tool_calls[1].function.arguments = '{"x": 4, "y": 2}'
            return result
        else:
            # Second call - stop
            result = AsyncMock()
            result.choices = [AsyncMock()]
            result.choices[0].finish_reason = "stop"
            result.choices[0].message.content = "Done"
            return result

    mock_client.chat.completions.create.side_effect = side_effect

    agent = AsyncAgent(model="gpt-4", client=mock_client, tools=[add_async, multiply_async])

    # Test
    messages = [{"role": "user", "content": "Do some math"}]
    result = await agent.run(messages)

    # Assertions
    assert result.choices[0].finish_reason == "stop"
    assert mock_client.chat.completions.create.call_count == 2
    assert len(agent.tools) == 2
    assert "add_async" in agent.tool_map
    assert "multiply_async" in agent.tool_map


@pytest.mark.asyncio
async def test_async_agent_with_mixed_sync_async_tools():
    @tool
    async def add_async(a: int, b: int) -> int:
        """Add two numbers asynchronously."""
        await asyncio.sleep(0.001)
        return a + b

    @tool
    def subtract_sync(a: int, b: int) -> int:
        """Subtract two numbers synchronously."""
        return a - b

    # Mock client
    mock_client = AsyncMock()

    def side_effect(*args, **kwargs):
        if mock_client.chat.completions.create.call_count == 1:
            # First call - tool calls
            result = AsyncMock()
            result.choices = [AsyncMock()]
            result.choices[0].finish_reason = "tool_calls"
            result.choices[0].message.tool_calls = [
                AsyncMock(id="call_1", function=AsyncMock(name="add_async", arguments='{"a": 10, "b": 5}')),
                AsyncMock(id="call_2", function=AsyncMock(name="subtract_sync", arguments='{"a": 20, "b": 8}')),
            ]
            result.choices[0].message.tool_calls[0].function.name = "add_async"
            result.choices[0].message.tool_calls[0].function.arguments = '{"a": 10, "b": 5}'
            result.choices[0].message.tool_calls[1].function.name = "subtract_sync"
            result.choices[0].message.tool_calls[1].function.arguments = '{"a": 20, "b": 8}'
            return result
        else:
            # Second call - stop
            result = AsyncMock()
            result.choices = [AsyncMock()]
            result.choices[0].finish_reason = "stop"
            result.choices[0].message.content = "Mixed calculations done"
            return result

    mock_client.chat.completions.create.side_effect = side_effect

    # Create agent
    agent = AsyncAgent(model="gpt-4", client=mock_client, tools=[add_async, subtract_sync])

    # Test
    messages = [{"role": "user", "content": "Do mixed calculations"}]
    result = await agent.run(messages)

    # Assertions
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.content == "Mixed calculations done"
    assert mock_client.chat.completions.create.call_count == 2
    assert len(agent.tools) == 2
    assert "add_async" in agent.tool_map
    assert "subtract_sync" in agent.tool_map

    # Verify both sync and async tools are properly handled
    calls = mock_client.chat.completions.create.call_args_list
    final_messages = calls[1][1]["messages"]

    # Should have tool responses
    tool_responses = [msg for msg in final_messages if msg.get("role") == "tool"]
    assert len(tool_responses) == 2

    # Parse content values robustly (JSON or raw) and compare without assuming order
    def _to_int(value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                pass
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(value)
        except Exception:
            pytest.fail(f"Unexpected tool content format: {value!r}")

    contents = {_to_int(msg["content"]) for msg in tool_responses}
    assert contents == {15, 12}


@pytest.mark.asyncio
async def test_async_agent_tool_error_handling():
    @tool
    def divide_with_error(a: int, b: int) -> float:
        """Divide with potential error."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    # Mock client
    mock_client = AsyncMock()

    def side_effect(*args, **kwargs):
        if mock_client.chat.completions.create.call_count == 1:
            result = AsyncMock()
            result.choices = [AsyncMock()]
            result.choices[0].finish_reason = "tool_calls"
            result.choices[0].message.tool_calls = [
                AsyncMock(id="call_1", function=AsyncMock(name="divide_with_error", arguments='{"a": 10, "b": 0}'))
            ]
            result.choices[0].message.tool_calls[0].function.name = "divide_with_error"
            result.choices[0].message.tool_calls[0].function.arguments = '{"a": 10, "b": 0}'
            return result
        else:
            result = AsyncMock()
            result.choices = [AsyncMock()]
            result.choices[0].finish_reason = "stop"
            result.choices[0].message.content = "Error handled"
            return result

    mock_client.chat.completions.create.side_effect = side_effect

    # Create agent
    agent = AsyncAgent(model="gpt-4", client=mock_client, tools=[divide_with_error])

    # Test
    messages = [{"role": "user", "content": "Divide by zero"}]
    result = await agent.run(messages)

    # Assertions
    assert result.choices[0].finish_reason == "stop"
    assert mock_client.chat.completions.create.call_count == 2

    # Check error was passed to model
    calls = mock_client.chat.completions.create.call_args_list
    final_messages = calls[1][1]["messages"]
    tool_response = next(msg for msg in final_messages if msg.get("role") == "tool")
    assert "Cannot divide by zero" in tool_response["content"]


def test_async_agent_duplicate_tool_names():
    @tool
    def duplicate_name(x: int) -> int:
        return x

    @tool
    def duplicate_name_2(y: int) -> int:
        return y * 2

    # Manually set the second function to have the same name as the first
    duplicate_name_2._tool_schema["function"]["name"] = "duplicate_name"

    mock_client = AsyncMock()

    with pytest.raises(ValueError, match=r"Duplicate.*tool name"):
        AsyncAgent(model="gpt-4", client=mock_client, tools=[duplicate_name, duplicate_name_2])


def test_async_agent_undecorated_tool():
    def undecorated_tool(x: int) -> int:
        return x

    mock_client = AsyncMock()

    with pytest.raises(ValueError, match=r"not decorated.*@tool"):
        AsyncAgent(model="gpt-4", client=mock_client, tools=[undecorated_tool])
