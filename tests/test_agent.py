import asyncio
from dataclasses import dataclass
from typing import Any, Iterator, cast
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletionUserMessageParam

from zipcoil import Agent, tool


def test_agent_rejects_async_tools():
    @tool
    def sync_tool(x: int) -> int:
        return x

    @tool
    async def async_tool(x: int) -> int:
        return x

    mock_client = Mock()

    with pytest.raises(ValueError, match=r"async_tool.*?an async function"):
        Agent(model="gpt-4", client=mock_client, tools=[sync_tool, async_tool])


@dataclass
class _FakeContentDeltaEvent:
    type: str
    delta: str


@dataclass
class _FakeToolCallFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeToolCallFunction


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None


@dataclass
class _FakeChoice:
    finish_reason: str
    message: _FakeMessage


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]


class _FakeStreamManager:
    def __init__(self, events: list[_FakeContentDeltaEvent], completion: _FakeCompletion) -> None:
        self._events = events
        self._completion = completion

    def __enter__(self) -> "_FakeStreamManager":
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        return None

    def __iter__(self) -> Iterator[_FakeContentDeltaEvent]:
        for event in self._events:
            yield event

    def get_final_completion(self) -> _FakeCompletion:
        return self._completion


class _FakeCompletions:
    def __init__(self, stream_payloads: list[tuple[list[_FakeContentDeltaEvent], _FakeCompletion]]) -> None:
        self._stream_payloads = stream_payloads
        self.stream_calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStreamManager:
        self.stream_calls.append(kwargs)
        payload_index = len(self.stream_calls) - 1
        events, completion = self._stream_payloads[payload_index]
        return _FakeStreamManager(events=events, completion=completion)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, stream_payloads: list[tuple[list[_FakeContentDeltaEvent], _FakeCompletion]]) -> None:
        self.chat = _FakeChat(completions=_FakeCompletions(stream_payloads=stream_payloads))


def _make_tool_call(tool_call_id: str, name: str, arguments: str) -> _FakeToolCall:
    return _FakeToolCall(id=tool_call_id, function=_FakeToolCallFunction(name=name, arguments=arguments))


def _make_completion(
    finish_reason: str,
    *,
    content: str | None = None,
    tool_calls: list[_FakeToolCall] | None = None,
) -> _FakeCompletion:
    return _FakeCompletion(choices=[_FakeChoice(finish_reason=finish_reason, message=_FakeMessage(content, tool_calls))])


def test_agent_streaming_with_tools() -> None:
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    stream_payloads = [
        (
            [_FakeContentDeltaEvent(type="content.delta", delta="Calculating")],
            _make_completion(
                "tool_calls",
                tool_calls=[_make_tool_call("call_1", "add", '{"a": 2, "b": 3}')],
            ),
        ),
        (
            [_FakeContentDeltaEvent(type="content.delta", delta="Done")],
            _make_completion("stop", content="Final response"),
        ),
    ]
    fake_client = _FakeClient(stream_payloads=stream_payloads)

    agent = Agent(model="gpt-4", client=cast(Any, fake_client), tools=[add])

    deltas: list[str] = []
    event_types: list[str] = []
    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Please add 2 and 3"})]

    result = agent.run(
        messages=messages,
        stream=True,
        on_stream_event=lambda event: event_types.append(event.type),
        on_text_delta=deltas.append,
    )

    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.content == "Final response"
    assert "".join(deltas) == "CalculatingDone"
    assert event_types == ["content.delta", "content.delta"]
    assert len(fake_client.chat.completions.stream_calls) == 2

    second_call_messages = fake_client.chat.completions.stream_calls[1]["messages"]
    tool_messages = [msg for msg in second_call_messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "5"


def test_agent_streaming_supports_async_callbacks() -> None:
    stream_payloads = [
        (
            [
                _FakeContentDeltaEvent(type="content.delta", delta="Hello"),
                _FakeContentDeltaEvent(type="content.delta", delta="!"),
            ],
            _make_completion("stop", content="Done"),
        )
    ]
    fake_client = _FakeClient(stream_payloads=stream_payloads)
    agent = Agent(model="gpt-4", client=cast(Any, fake_client), tools=[])

    deltas: list[str] = []
    event_types: list[str] = []
    loop_ids: list[int] = []

    async def on_stream_event(event: Any) -> None:
        await asyncio.sleep(0)
        event_types.append(event.type)

    async def on_text_delta(delta: str) -> None:
        await asyncio.sleep(0)
        loop_ids.append(id(asyncio.get_running_loop()))
        deltas.append(delta)

    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Say hi"})]
    result = agent.run(
        messages=messages,
        stream=True,
        on_stream_event=on_stream_event,
        on_text_delta=on_text_delta,
    )

    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.content == "Done"
    assert event_types == ["content.delta", "content.delta"]
    assert deltas == ["Hello", "!"]
    assert len(set(loop_ids)) == 1


@pytest.mark.asyncio
async def test_agent_sync_callback_raises_when_event_loop_is_running() -> None:
    async def _async_callback() -> None:
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="event loop is already running"):
        Agent._run_sync_callback(_async_callback(), cast(Any, None))
