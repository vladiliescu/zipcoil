import json
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk

from zipcoil import Agent, AsyncAgent, tool


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    return (
        "data: "
        + json.dumps(
            {
                "id": "completion_1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )
        + "\n\n"
    )


@pytest.fixture
def transport() -> httpx.MockTransport:
    # Exercise the real SDK's validation and stream parsing; only HTTP responses are simulated.
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        if body["tools"] and body["messages"][-1]["role"] != "tool":
            content = _chunk(
                {
                    "role": "assistant",
                    "content": "Calculating",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a": 2,'},
                        }
                    ],
                }
            )
            content += _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"b": 3}'}}]})
            content += _chunk({}, "tool_calls")
        else:
            if body["tools"]:
                assert body["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "5"}
            content = _chunk({"role": "assistant", "content": "Done"}, "stop")
        annotation = {
            "object": "",
            "choices": [{"index": 0, "finish_reason": None, "content_filter_results": {}}],
        }
        content += "data: " + json.dumps(annotation) + "\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content + "data: [DONE]\n\n")

    return httpx.MockTransport(respond)


@pytest.mark.parametrize("strict", [True, False, None], ids=["strict", "non-strict", "no-tools"])
def test_agent_streaming(transport: httpx.MockTransport, strict: bool | None) -> None:
    received: list[tuple[int, int]] = []

    @tool(strict=bool(strict))
    def add(a: int, b: int) -> int:
        received.append((a, b))
        return a + b

    with OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport)) as client:
        agent = Agent(model="gpt-4o", client=client, tools=[add] if strict is not None else [])
        chunks = list(agent.run(messages=[{"role": "user", "content": "Add 2 and 3."}], stream=True))

    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert received == ([(2, 3)] if strict is not None else [])
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
        "CalculatingDone" if strict is not None else "Done"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [True, False, None], ids=["strict", "non-strict", "no-tools"])
async def test_async_agent_streaming(transport: httpx.MockTransport, strict: bool | None) -> None:
    received: list[tuple[int, int]] = []

    @tool(strict=bool(strict))
    async def add(a: int, b: int) -> int:
        received.append((a, b))
        return a + b

    async with AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=transport)) as client:
        agent = AsyncAgent(model="gpt-4o", client=client, tools=[add] if strict is not None else [])
        stream = await agent.run(messages=[{"role": "user", "content": "Add 2 and 3."}], stream=True)
        chunks = [chunk async for chunk in stream]

    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert received == ([(2, 3)] if strict is not None else [])
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
        "CalculatingDone" if strict is not None else "Done"
    )
