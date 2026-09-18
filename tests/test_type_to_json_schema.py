from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pytest

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

    def test_basic_types(self) -> None:
        """Test conversion of basic Python types."""

        assert _type_to_json_schema(str) == {"type": "string"}
        assert _type_to_json_schema(int) == {"type": "integer"}
        assert _type_to_json_schema(float) == {"type": "number"}
        assert _type_to_json_schema(bool) == {"type": "boolean"}
        assert _type_to_json_schema(list) == {"type": "array", "items": {}}
        assert _type_to_json_schema(dict) == {"type": "object", "additionalProperties": {}}
        assert _type_to_json_schema(Color) == {"type": "string", "enum": ["red", "green", "blue"]}
        assert _type_to_json_schema(Mixed) == {"type": "string", "enum": ["1", "second", "False"]}

    def test_generic_types(self) -> None:
        """Test conversion of generic types."""
        assert _type_to_json_schema(List[str]) == {"type": "array", "items": {"type": "string"}}
        assert _type_to_json_schema(Dict[str, int]) == {"type": "object", "additionalProperties": {"type": "integer"}}

    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (Any, {}),
            (list[Any], {"type": "array", "items": {}}),
            (dict[str, Any], {"type": "object", "additionalProperties": {}}),
            (int | Any, {"anyOf": [{"type": "integer"}, {}]}),
        ],
    )
    def test_any(self, annotation: Any, expected: dict[str, Any]) -> None:
        assert _type_to_json_schema(annotation) == expected

    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (List, {"type": "array", "items": {}}),
            (Dict, {"type": "object", "additionalProperties": {}}),
            (dict[str, int], {"type": "object", "additionalProperties": {"type": "integer"}}),
            (
                dict[str, list[int | float]],
                {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "integer"}, {"type": "number"}]},
                    },
                },
            ),
            (
                list[dict[str, int]],
                {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "integer"}}},
            ),
            (
                dict[str, int] | None,
                {"anyOf": [{"type": "object", "additionalProperties": {"type": "integer"}}, {"type": "null"}]},
            ),
        ],
    )
    def test_dictionaries_and_bare_containers(self, annotation: Any, expected: dict[str, Any]) -> None:
        assert _type_to_json_schema(annotation) == expected

    @pytest.mark.parametrize(
        ("annotation", "value_schema"),
        [
            (Optional[str], {"type": "string"}),
            (str | None, {"type": "string"}),
            (Optional[int], {"type": "integer"}),
            (int | None, {"type": "integer"}),
            (Optional[bool], {"type": "boolean"}),
            (bool | None, {"type": "boolean"}),
            (Optional[Color], {"type": "string", "enum": ["red", "green", "blue"]}),
            (Color | None, {"type": "string", "enum": ["red", "green", "blue"]}),
            (list[int] | None, {"type": "array", "items": {"type": "integer"}}),
        ],
    )
    def test_optional_types(self, annotation: Any, value_schema: dict[str, Any]) -> None:
        assert _type_to_json_schema(annotation) == {"anyOf": [value_schema, {"type": "null"}]}

    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (list[int], {"type": "array", "items": {"type": "integer"}}),
            (
                list[list[int]],
                {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
            ),
            (int | float, {"anyOf": [{"type": "integer"}, {"type": "number"}]}),
            (Union[int, float], {"anyOf": [{"type": "integer"}, {"type": "number"}]}),
            (
                int | float | None,
                {"anyOf": [{"type": "integer"}, {"type": "number"}, {"type": "null"}]},
            ),
            (
                list[int | float],
                {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "number"}]}},
            ),
            (
                list[int] | str,
                {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "string"}]},
            ),
        ],
    )
    def test_lists_and_unions(self, annotation: Any, expected: dict[str, Any]) -> None:
        assert _type_to_json_schema(annotation) == expected

    def test_unknown_types(self):
        """Test that unknown types default to string."""

        class CustomType:
            pass

        assert _type_to_json_schema(CustomType) == {"type": "string"}
