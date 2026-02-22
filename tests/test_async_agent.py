import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, cast
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletionUserMessageParam

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
    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Do some math"})]
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
            result = Mock()
            result.choices = [Mock()]
            result.choices[0].finish_reason = "tool_calls"
            result.choices[0].message.tool_calls = [
                Mock(id="call_1", function=Mock(name="add_async", arguments='{"a": 10, "b": 5}')),
                Mock(id="call_2", function=Mock(name="subtract_sync", arguments='{"a": 20, "b": 8}')),
            ]
            result.choices[0].message.tool_calls[0].function.name = "add_async"
            result.choices[0].message.tool_calls[0].function.arguments = '{"a": 10, "b": 5}'
            result.choices[0].message.tool_calls[1].function.name = "subtract_sync"
            result.choices[0].message.tool_calls[1].function.arguments = '{"a": 20, "b": 8}'
            return result
        else:
            # Second call - stop
            result = Mock()
            result.choices = [Mock()]
            result.choices[0].finish_reason = "stop"
            result.choices[0].message.content = "Mixed calculations done"
            return result

    mock_client.chat.completions.create.side_effect = side_effect

    # Create agent
    agent = AsyncAgent(model="gpt-4", client=mock_client, tools=[add_async, subtract_sync])

    # Test
    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Do mixed calculations"})]
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
            result = Mock()
            result.choices = [Mock()]
            result.choices[0].finish_reason = "tool_calls"
            result.choices[0].message.tool_calls = [
                Mock(id="call_1", function=Mock(name="divide_with_error", arguments='{"a": 10, "b": 0}'))
            ]
            result.choices[0].message.tool_calls[0].function.name = "divide_with_error"
            result.choices[0].message.tool_calls[0].function.arguments = '{"a": 10, "b": 0}'
            return result
        else:
            result = Mock()
            result.choices = [Mock()]
            result.choices[0].finish_reason = "stop"
            result.choices[0].message.content = "Error handled"
            return result

    mock_client.chat.completions.create.side_effect = side_effect

    # Create agent
    agent = AsyncAgent(model="gpt-4", client=mock_client, tools=[divide_with_error])

    # Test
    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Divide by zero"})]
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
    duplicate_name_2.tool_schema["function"]["name"] = "duplicate_name"

    mock_client = AsyncMock()

    with pytest.raises(ValueError, match=r"Duplicate.*tool name"):
        AsyncAgent(model="gpt-4", client=mock_client, tools=[duplicate_name, duplicate_name_2])


def test_async_agent_undecorated_tool():
    def undecorated_tool(x: int) -> int:
        return x

    mock_client = AsyncMock()

    with pytest.raises(ValueError, match=r"not decorated.*@tool"):
        # Intentionally testing error case with wrong type
        AsyncAgent(model="gpt-4", client=mock_client, tools=[undecorated_tool])  # type: ignore[list-item]


@dataclass
class _FakeAsyncContentDeltaEvent:
    type: str
    delta: str


@dataclass
class _FakeAsyncToolCallFunction:
    name: str
    arguments: str


@dataclass
class _FakeAsyncToolCall:
    id: str
    function: _FakeAsyncToolCallFunction


@dataclass
class _FakeAsyncMessage:
    content: str | None
    tool_calls: list[_FakeAsyncToolCall] | None


@dataclass
class _FakeAsyncChoice:
    finish_reason: str
    message: _FakeAsyncMessage


@dataclass
class _FakeAsyncCompletion:
    choices: list[_FakeAsyncChoice]


class _FakeAsyncStreamManager:
    def __init__(self, events: list[_FakeAsyncContentDeltaEvent], completion: _FakeAsyncCompletion) -> None:
        self._events = events
        self._completion = completion

    async def __aenter__(self) -> "_FakeAsyncStreamManager":
        return self

    async def __aexit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[_FakeAsyncContentDeltaEvent]:
        for event in self._events:
            yield event

    async def get_final_completion(self) -> _FakeAsyncCompletion:
        return self._completion


class _FakeAsyncCompletions:
    def __init__(self, stream_payloads: list[tuple[list[_FakeAsyncContentDeltaEvent], _FakeAsyncCompletion]]) -> None:
        self._stream_payloads = stream_payloads
        self.stream_calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeAsyncStreamManager:
        self.stream_calls.append(kwargs)
        payload_index = len(self.stream_calls) - 1
        events, completion = self._stream_payloads[payload_index]
        return _FakeAsyncStreamManager(events=events, completion=completion)


class _FakeAsyncChat:
    def __init__(self, completions: _FakeAsyncCompletions) -> None:
        self.completions = completions


class _FakeAsyncClient:
    def __init__(self, stream_payloads: list[tuple[list[_FakeAsyncContentDeltaEvent], _FakeAsyncCompletion]]) -> None:
        self.chat = _FakeAsyncChat(completions=_FakeAsyncCompletions(stream_payloads=stream_payloads))


def _make_async_tool_call(tool_call_id: str, name: str, arguments: str) -> _FakeAsyncToolCall:
    return _FakeAsyncToolCall(id=tool_call_id, function=_FakeAsyncToolCallFunction(name=name, arguments=arguments))


def _make_async_completion(
    finish_reason: str,
    *,
    content: str | None = None,
    tool_calls: list[_FakeAsyncToolCall] | None = None,
) -> _FakeAsyncCompletion:
    return _FakeAsyncCompletion(
        choices=[_FakeAsyncChoice(finish_reason=finish_reason, message=_FakeAsyncMessage(content, tool_calls))]
    )


@pytest.mark.asyncio
async def test_async_agent_streaming_with_async_callbacks() -> None:
    @tool
    async def add_async(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b

    stream_payloads = [
        (
            [_FakeAsyncContentDeltaEvent(type="content.delta", delta="Working")],
            _make_async_completion(
                "tool_calls",
                tool_calls=[_make_async_tool_call("call_1", "add_async", '{"a": 4, "b": 5}')],
            ),
        ),
        (
            [_FakeAsyncContentDeltaEvent(type="content.delta", delta="Done")],
            _make_async_completion("stop", content="Async final response"),
        ),
    ]
    fake_client = _FakeAsyncClient(stream_payloads=stream_payloads)
    agent = AsyncAgent(model="gpt-4", client=cast(Any, fake_client), tools=[add_async])

    events_seen: list[str] = []
    deltas: list[str] = []

    async def on_stream_event(event: Any) -> None:
        events_seen.append(event.type)

    async def on_text_delta(delta: str) -> None:
        await asyncio.sleep(0)
        deltas.append(delta)

    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Please add 4 and 5"})]
    result = await agent.run(
        messages=messages,
        stream=True,
        on_stream_event=on_stream_event,
        on_text_delta=on_text_delta,
    )

    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.content == "Async final response"
    assert "".join(deltas) == "WorkingDone"
    assert events_seen == ["content.delta", "content.delta"]
    assert len(fake_client.chat.completions.stream_calls) == 2

    second_call_messages = fake_client.chat.completions.stream_calls[1]["messages"]
    tool_messages = [msg for msg in second_call_messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "9"


@pytest.mark.asyncio
async def test_async_agent_streaming_accepts_sync_callbacks() -> None:
    stream_payloads = [
        (
            [_FakeAsyncContentDeltaEvent(type="content.delta", delta="Hello")],
            _make_async_completion("stop", content="Sync callback response"),
        )
    ]
    fake_client = _FakeAsyncClient(stream_payloads=stream_payloads)
    agent = AsyncAgent(model="gpt-4", client=cast(Any, fake_client), tools=[])

    event_types: list[str] = []
    deltas: list[str] = []

    def on_stream_event(event: Any) -> None:
        event_types.append(event.type)

    def on_text_delta(delta: str) -> None:
        deltas.append(delta)

    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Say hi"})]
    result = await agent.run(
        messages=messages,
        stream=True,
        on_stream_event=on_stream_event,
        on_text_delta=on_text_delta,
    )

    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.content == "Sync callback response"
    assert event_types == ["content.delta"]
    assert deltas == ["Hello"]
