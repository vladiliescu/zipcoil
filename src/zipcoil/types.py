from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ToolProtocol(Protocol):
    _tool_schema: Dict[str, Any]

    def __call__(self, **kwargs: Any) -> Any: ...
