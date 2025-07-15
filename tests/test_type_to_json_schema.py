from enum import Enum
from typing import List, Dict, Optional

from smith import _type_to_json_schema


class TestTypeToJsonSchema:
    """Test the _type_to_json_schema helper function."""

    def test_basic_types(self):
        """Test conversion of basic Python types."""

        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        assert _type_to_json_schema(str) == "string"
        assert _type_to_json_schema(int) == "integer"
        assert _type_to_json_schema(float) == "number"
        assert _type_to_json_schema(bool) == "boolean"
        assert _type_to_json_schema(list) == "array"
        assert _type_to_json_schema(dict) == "object"
        assert _type_to_json_schema(Color) == "enum: [red,green,blue]"

    def test_generic_types(self):
        """Test conversion of generic types."""
        assert _type_to_json_schema(List[str]) == "array"
        assert _type_to_json_schema(Dict[str, int]) == "object"

    def test_optional_types(self):
        """Test conversion of Optional types."""
        assert _type_to_json_schema(Optional[str]) == "string"
        assert _type_to_json_schema(Optional[int]) == "integer"
        assert _type_to_json_schema(Optional[bool]) == "boolean"

    def test_unknown_types(self):
        """Test that unknown types default to string."""

        class CustomType:
            pass

        assert _type_to_json_schema(CustomType) == "string"
