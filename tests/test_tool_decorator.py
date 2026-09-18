from enum import Enum
from typing import Any, Dict, List, Optional, assert_type, cast

import pytest
from dotenv import load_dotenv

from zipcoil import tool
from zipcoil.types import AsyncToolProtocol, ToolProtocol

load_dotenv()


class TestToolDecorator:
    """Test the @tool decorator functionality."""

    @pytest.mark.parametrize("strict", [True, False])
    def test_configured_sync_tool(self, strict: bool) -> None:
        @tool(strict=strict)
        def total(values: list[int]) -> int:
            """Add the supplied values."""
            return sum(values)

        assert_type(total, ToolProtocol)
        assert total(values=[2, 3]) == 5
        assert total.tool_schema["function"]["strict"] is strict

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strict", [True, False])
    async def test_configured_async_tool(self, strict: bool) -> None:
        @tool(strict=strict)
        async def total(values: list[int]) -> int:
            """Add the supplied values."""
            return sum(values)

        assert_type(total, AsyncToolProtocol)
        assert await total(values=[2, 3]) == 5
        assert total.tool_schema["function"]["strict"] is strict

    def test_parenthesized_decorator_defaults_to_strict(self) -> None:
        @tool()
        def echo(values: list[int]) -> list[int]:
            return values

        assert echo(values=[2, 3]) == [2, 3]
        assert echo.tool_schema["function"]["strict"] is True

    @pytest.mark.parametrize(
        ("annotation", "value"),
        [
            (Any, 7),
            (list, [2, "hello", None]),
            (List, [2, "hello", None]),
            (dict, {"apples": 2}),
            (Dict, {"apples": 2}),
            (dict[str, int], {"apples": 2}),
            (Dict[str, int], {"apples": 2}),
            (list[Any], [2, "hello", None]),
            (dict[str, Any], {"apples": 2}),
            (list[dict[str, int]], [{"apples": 2}]),
            (list[list], [[2, "hello"]]),
            (int | Any, None),
            (int | dict[str, int], {"apples": 2}),
            (Optional[List], None),
            (list[int | list[Any]], [2, ["hello"]]),
        ],
    )
    def test_inputs_requiring_non_strict_mode(self, annotation: Any, value: Any) -> None:
        def echo(value: annotation) -> Any:
            return value

        with pytest.raises(ValueError, match=r"Tool 'echo': parameter 'value'.*@tool\(strict=False\)"):
            tool(echo)

        decorated = tool(echo, strict=False)
        assert decorated(value=value) == value
        assert decorated.tool_schema["function"]["strict"] is False

    @pytest.mark.asyncio
    async def test_async_input_requires_non_strict_mode(self) -> None:
        async def echo(value: list[Any]) -> Any:
            return value

        with pytest.raises(ValueError, match=r"Tool 'echo': parameter 'value'.*@tool\(strict=False\)"):
            tool(strict=True)(echo)

        decorated = tool(strict=False)(echo)
        assert await decorated(value=[2, "hello"]) == [2, "hello"]

    @pytest.mark.parametrize("annotation", [list[int], list[list[int | float | None]], Optional[str]])
    def test_nested_strict_inputs(self, annotation: Any) -> None:
        @tool
        def echo(value: annotation) -> Any:
            return value

        assert echo.tool_schema["function"]["strict"] is True

    @pytest.mark.parametrize(
        ("annotation", "result"),
        [(Any, 7), (list, [2, "hello"]), (dict[str, Any], {"apples": 2})],
    )
    def test_strict_mode_does_not_restrict_return_types(self, annotation: Any, result: Any) -> None:
        @tool
        def get_result() -> annotation:
            return result

        assert get_result() == result
        assert get_result.tool_schema["function"]["strict"] is True

    def test_simple_function_with_single_required_arg(self):
        """Test a function with a single required string argument."""

        @tool
        def get_user(name: str) -> str:
            """Get user information by name.

            Args:
                name: The user's name
            """
            return f"User: {name}"

        schema = get_user.tool_schema

        assert schema["type"] == "function"
        function_def = schema["function"]
        assert function_def["name"] == "get_user"
        assert function_def.get("description") == "Get user information by name."

        params = function_def.get("parameters")
        assert params is not None
        params_dict = cast(Dict[str, Any], params)
        assert params_dict["type"] == "object"
        assert params_dict["required"] == ["name"]
        assert params_dict["additionalProperties"] is False
        assert function_def.get("strict") is True

        properties = cast(Dict[str, Any], params_dict["properties"])
        assert "name" in properties
        assert len(properties) == 1
        assert properties["name"]["type"] == "string"
        assert properties["name"]["description"] == "The user's name"

    def test_function_with_multiple_args_and_optional(self) -> None:
        """Test a function with multiple arguments including optional ones."""

        @tool
        def calculate(x: int, y: float, operation: str | None) -> float:
            """Perform a calculation on two numbers.

            Args:
                x: First number (integer)
                y: Second number (float)
                operation: Type of operation to perform
            """

            return 0.0

        schema = calculate.tool_schema

        function_def = schema["function"]
        assert function_def["name"] == "calculate"
        assert function_def.get("description") == "Perform a calculation on two numbers."

        params = function_def.get("parameters")
        assert params is not None
        params_dict = cast(Dict[str, Any], params)
        # operation has a default value, but still needs to be included here b/c strict=True
        assert set(params_dict["required"]) == {"x", "y", "operation"}

        properties = cast(Dict[str, Any], params_dict["properties"])
        assert properties["x"]["type"] == "integer"
        assert properties["x"]["description"] == "First number (integer)"
        assert properties["y"]["type"] == "number"
        assert properties["y"]["description"] == "Second number (float)"
        assert properties["operation"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
        assert properties["operation"]["description"] == "Type of operation to perform"

    def test_function_with_optional_type_hint(self) -> None:
        """Test a function with Optional type hints."""

        @tool
        def search(query: str, limit: Optional[int]) -> List[str]:
            """Search for items matching a query.

            Args:
                query: Search query string
                limit: Maximum number of results to return
            """
            return ["result1", "result2"]

        schema = search.tool_schema

        params = schema["function"].get("parameters")
        assert params is not None
        params_dict = cast(Dict[str, Any], params)
        assert params_dict["required"] == ["query", "limit"]

        properties = cast(Dict[str, Any], params_dict["properties"])
        assert properties["query"]["type"] == "string"
        assert properties["limit"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]

    def test_function_with_various_types(self):
        """Test a function with various Python types."""

        class Status(Enum):
            PENDING = 0
            SUCCESS = 1
            FAILURE = 2

        @tool(strict=False)
        def process_data(
            text: str,
            count: int,
            score: float,
            active: bool,
            tags: List[str],
            metadata: Dict[str, str | int | float | bool],
            status: Status,
        ) -> str:
            """Process data with various types.

            Args:
                text: Input text
                count: Number count
                score: Score value
                active: Whether active
                tags: List of tags
                metadata: Metadata dictionary
                status: Status of the process
            """
            return "processed"

        schema = process_data.tool_schema
        params = schema["function"].get("parameters")
        assert params is not None
        properties = cast(Dict[str, Any], params)["properties"]
        properties_dict = cast(Dict[str, Any], properties)

        assert properties_dict["text"]["type"] == "string"
        assert properties_dict["count"]["type"] == "integer"
        assert properties_dict["score"]["type"] == "number"
        assert properties_dict["active"]["type"] == "boolean"
        assert properties_dict["tags"]["type"] == "array"
        assert properties_dict["metadata"]["type"] == "object"
        assert properties_dict["status"]["type"] == "integer"
        assert properties_dict["status"].get("enum") == [0, 1, 2]

    def test_function_without_docstring(self):
        """Test a function without a docstring."""

        @tool
        def simple_func(arg: str) -> str:
            return arg.upper()

        schema = simple_func.tool_schema

        assert schema["function"].get("description") == ""

        # Should still extract parameter info
        params = schema["function"].get("parameters")
        assert params is not None
        properties = cast(Dict[str, Any], params)["properties"]
        properties_dict = cast(Dict[str, Any], properties)
        assert "arg" in properties_dict
        assert properties_dict["arg"]["type"] == "string"
        assert properties_dict["arg"]["description"] == ""

    def test_function_preserves_original_functionality(self):
        """Test that the decorated function still works normally."""

        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number
                b: Second number
            """
            return a + b

        result = add_numbers(a=5, b=3)
        assert result == 8

        assert hasattr(add_numbers, "tool_schema")
