from enum import Enum
from typing import Dict, List, Optional

import pytest
from dotenv import load_dotenv

from zipcoil import tool

load_dotenv()


class TestToolDecorator:
    """Test the @tool decorator functionality."""

    def test_simple_function_with_single_required_arg(self):
        """Test a function with a single required string argument."""

        @tool
        def get_user(name: str) -> str:
            """Get user information by name.

            Args:
                name: The user's name
            """
            return f"User: {name}"

        schema = get_user._tool_schema

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_user"
        assert schema["function"]["description"] == "Get user information by name."

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert params["required"] == ["name"]
        assert params["additionalProperties"] is False
        assert schema["function"]["strict"] is True

        properties = params["properties"]
        assert "name" in properties
        assert len(properties) == 1
        assert properties["name"]["type"] == "string"
        assert properties["name"]["description"] == "The user's name"

    def test_function_with_multiple_args_and_optional(self):
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

        schema = calculate._tool_schema

        assert schema["function"]["name"] == "calculate"
        assert schema["function"]["description"] == "Perform a calculation on two numbers."

        params = schema["function"]["parameters"]
        # operation has a default value, but still needs to be included here b/c strict=True
        assert set(params["required"]) == {"x", "y", "operation"}

        properties = params["properties"]
        assert properties["x"]["type"] == "integer"
        assert properties["x"]["description"] == "First number (integer)"
        assert properties["y"]["type"] == "number"
        assert properties["y"]["description"] == "Second number (float)"
        assert properties["operation"]["type"] == ["string", "null"]
        assert properties["operation"]["description"] == "Type of operation to perform"

    def test_function_with_optional_type_hint(self):
        """Test a function with Optional type hints."""

        @tool
        def search(query: str, limit: Optional[int]) -> List[str]:
            """Search for items matching a query.

            Args:
                query: Search query string
                limit: Maximum number of results to return
            """
            return ["result1", "result2"]

        schema = search._tool_schema

        params = schema["function"]["parameters"]
        assert params["required"] == ["query", "limit"]

        properties = params["properties"]
        assert properties["query"]["type"] == "string"
        assert properties["limit"]["type"] == ["integer", "null"]

    def test_function_with_various_types(self):
        """Test a function with various Python types."""

        class Status(Enum):
            PENDING = 0
            SUCCESS = 1
            FAILURE = 2

        @tool
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

        schema = process_data._tool_schema
        properties = schema["function"]["parameters"]["properties"]

        assert properties["text"]["type"] == "string"
        assert properties["count"]["type"] == "integer"
        assert properties["score"]["type"] == "number"
        assert properties["active"]["type"] == "boolean"
        assert properties["tags"]["type"] == "array"
        assert properties["metadata"]["type"] == "object"
        assert properties["status"]["type"] == "integer"
        assert properties["status"]["enum"] == [0, 1, 2]

    def test_function_without_docstring(self):
        """Test a function without a docstring."""

        @tool
        def simple_func(arg: str) -> str:
            return arg.upper()

        schema = simple_func._tool_schema

        assert schema["function"]["description"] == ""

        # Should still extract parameter info
        properties = schema["function"]["parameters"]["properties"]
        assert "arg" in properties
        assert properties["arg"]["type"] == "string"
        assert properties["arg"]["description"] == ""

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

        result = add_numbers(5, 3)
        assert result == 8

        assert hasattr(add_numbers, "_tool_schema")

    def test_async_tool_rejection(self):
        """Test that async tools are rejected during decoration."""

        # The error should be raised when applying the @tool decorator
        with pytest.raises(ValueError, match="Async tools are not supported"):

            @tool
            async def async_get_weather(city: str) -> str:
                """Get weather information for a city.

                Args:
                    city: The city name
                """
                return f"Weather in {city}: sunny"
