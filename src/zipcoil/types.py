from typing import Any, Protocol, runtime_checkable

from openai.types.chat import ChatCompletionToolParam


@runtime_checkable
class ToolProtocol(Protocol):
    tool_schema: ChatCompletionToolParam

    def __call__(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class AsyncToolProtocol(Protocol):
    tool_schema: ChatCompletionToolParam

    async def __call__(self, **kwargs: Any) -> Any: ...
