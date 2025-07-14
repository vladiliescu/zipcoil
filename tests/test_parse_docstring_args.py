from smith import _parse_docstring_args


class TestParseDocstringArgs:
    """Test the _parse_docstring_args helper function."""

    def test_simple_args_section(self):
        docstring = """Function description.

        Args:
            arg1: First argument
            arg2: Second argument
        """

        result = _parse_docstring_args(docstring)
        assert result == {
            "arg1": "First argument",
            "arg2": "Second argument"
        }

    def test_args_with_multiline_descriptions(self):
        docstring = """Function description.

        Args:
            arg1: This is a long description
                  that spans multiple lines
            arg2: Short description
        """

        result = _parse_docstring_args(docstring)
        assert "arg1" in result
        assert result["arg1"] == """This is a long description
                  that spans multiple lines"""
        assert "arg2" in result

    def test_no_args_section(self):
        docstring = """Function description without args."""

        result = _parse_docstring_args(docstring)
        assert result == {}

    def test_empty_docstring(self):
        assert _parse_docstring_args(None) == {}
        assert _parse_docstring_args("") == {}

    def test_args_with_returns_section(self):
        docstring = """Function description.

        Args:
            arg1: First argument
            arg2: Second argument

        Returns:
            Some return value
        """

        result = _parse_docstring_args(docstring)
        assert result == {
            "arg1": "First argument",
            "arg2": "Second argument"
        }

    def test_malformed_args_section(self):
        docstring = """Function description.

        Args:
            arg1 - Missing colon
            arg2: Valid argument
            : Missing argument name
        """

        result = _parse_docstring_args(docstring)
        # Should only parse valid lines
        assert "arg2" in result
        assert result["arg2"] == "Valid argument"
