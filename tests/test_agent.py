from dataclasses import dataclass
from typing import Any, Iterator, cast
from unittest.mock import Mock

import pytest
from openai.types.chat import ChatCompletionChunk, ChatCompletionUserMessageParam

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


@dataclass
class _FakeChunkEvent:
    chunk: ChatCompletionChunk
    type: str = "chunk"


@dataclass
class _FakeNonChunkEvent:
    type: str


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
    def __init__(self, events: list[Any], completion: _FakeCompletion) -> None:
        self._events = events
        self._completion = completion

    def __enter__(self) -> "_FakeStreamManager":
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        return None

    def __iter__(self) -> Iterator[Any]:
        for event in self._events:
            yield event

    def get_final_completion(self) -> _FakeCompletion:
        return self._completion


class _FakeCompletions:
    def __init__(self, stream_payloads: list[tuple[list[Any], _FakeCompletion]]) -> None:
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
    def __init__(self, stream_payloads: list[tuple[list[Any], _FakeCompletion]]) -> None:
        self.chat = _FakeChat(completions=_FakeCompletions(stream_payloads=stream_payloads))


def _make_chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk_1",
        object="chat.completion.chunk",
        created=0,
        model="gpt-4",
        choices=cast(
            Any,
            [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        ),
    )


def _make_tool_call(tool_call_id: str, name: str, arguments: str) -> _FakeToolCall:
    return _FakeToolCall(id=tool_call_id, function=_FakeToolCallFunction(name=name, arguments=arguments))


def _make_completion(
    finish_reason: str,
    *,
    content: str | None = None,
    tool_calls: list[_FakeToolCall] | None = None,
) -> _FakeCompletion:
    return _FakeCompletion(choices=[_FakeChoice(finish_reason=finish_reason, message=_FakeMessage(content, tool_calls))])


def test_agent_streaming_with_tools_returns_chunks() -> None:
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    stream_payloads = [
        (
            [
                _FakeChunkEvent(chunk=_make_chunk("Calculating")),
                _FakeNonChunkEvent(type="content.delta"),
            ],
            _make_completion(
                "tool_calls",
                tool_calls=[_make_tool_call("call_1", "add", '{"a": 2, "b": 3}')],
            ),
        ),
        (
            [_FakeChunkEvent(chunk=_make_chunk("Done"))],
            _make_completion("stop", content="Final response"),
        ),
    ]
    fake_client = _FakeClient(stream_payloads=stream_payloads)

    agent = Agent(model="gpt-4", client=cast(Any, fake_client), tools=[add])

    messages = [cast(ChatCompletionUserMessageParam, {"role": "user", "content": "Please add 2 and 3"})]
    stream = agent.run(messages=messages, stream=True)

    chunks = list(cast(Iterator[ChatCompletionChunk], stream))
    content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)

    assert content == "CalculatingDone"
    assert len(fake_client.chat.completions.stream_calls) == 2

    second_call_messages = fake_client.chat.completions.stream_calls[1]["messages"]
    tool_messages = [msg for msg in second_call_messages if isinstance(msg, dict) and msg.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "5"


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
