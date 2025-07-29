from enum import Enum
from typing import Dict, List, Optional

from zipcoil.core import _type_to_json_schema


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Mixed(Enum):
    First = 1
    Second = "second"
    Third = False


class TestTypeToJsonSchema:
    """Test the _type_to_json_schema helper function."""

    def test_basic_types(self):
        """Test conversion of basic Python types."""

        assert _type_to_json_schema(str) == {"type": "string"}
        assert _type_to_json_schema(int) == {"type": "integer"}
        assert _type_to_json_schema(float) == {"type": "number"}
        assert _type_to_json_schema(bool) == {"type": "boolean"}
        assert _type_to_json_schema(list) == {"type": "array"}
        assert _type_to_json_schema(dict) == {"type": "object"}
        assert _type_to_json_schema(Color) == {"type": "string", "enum": ["red", "green", "blue"]}
        assert _type_to_json_schema(Mixed) == {"type": "string", "enum": ["1", "second", "False"]}

    def test_generic_types(self):
        """Test conversion of generic types."""
        assert _type_to_json_schema(List[str]) == {"type": "array"}
        assert _type_to_json_schema(Dict[str, int]) == {"type": "object"}

    def test_optional_types(self):
        """Test conversion of Optional types."""
        assert _type_to_json_schema(Optional[str]) == {"type": ["string", "null"]}
        assert _type_to_json_schema(Optional[int]) == {"type": ["integer", "null"]}
        assert _type_to_json_schema(Optional[bool]) == {"type": ["boolean", "null"]}
        assert _type_to_json_schema(Optional[Color]) == {
            "type": ["string", "null"],
            "enum": ["red", "green", "blue"],
        }

    def test_new_union_syntax(self):
        """Test conversion of new union syntax (T | None)."""
        assert _type_to_json_schema(str | None) == {"type": ["string", "null"]}
        assert _type_to_json_schema(int | None) == {"type": ["integer", "null"]}
        assert _type_to_json_schema(bool | None) == {"type": ["boolean", "null"]}
        assert _type_to_json_schema(Color | None) == {
            "type": ["string", "null"],
            "enum": ["red", "green", "blue"],
        }

    def test_unknown_types(self):
        """Test that unknown types default to string."""

        class CustomType:
            pass

        assert _type_to_json_schema(CustomType) == {"type": "string"}
