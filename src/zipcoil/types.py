from typing import Any, Awaitable, Callable, Protocol, overload, runtime_checkable

from openai.types.chat import ChatCompletionToolParam


@runtime_checkable
class ToolProtocol(Protocol):
    tool_schema: ChatCompletionToolParam

    def __call__(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class AsyncToolProtocol(Protocol):
    tool_schema: ChatCompletionToolParam

    async def __call__(self, **kwargs: Any) -> Any: ...


class ToolDecoratorProtocol(Protocol):
    @overload
    def __call__(self, func: Callable[..., Awaitable[Any]]) -> AsyncToolProtocol: ...

    @overload
    def __call__(self, func: Callable[..., Any]) -> ToolProtocol: ...
