from enum import Enum
from typing import Any, Dict, List, Optional, cast

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

        schema = search.tool_schema

        params = schema["function"].get("parameters")
        assert params is not None
        params_dict = cast(Dict[str, Any], params)
        assert params_dict["required"] == ["query", "limit"]

        properties = cast(Dict[str, Any], params_dict["properties"])
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
