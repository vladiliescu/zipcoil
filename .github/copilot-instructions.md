## About

We're building Zipcoil, a library for simplifying OpenAI tool usage, helping users build simple AI agents using the OpenAI library.

## Functionality

- When implementing new functionalities, use standard built-in libraries as much as possible. If work can be simplified using an existing external library check with me first. Consider the tradeoffs between writing (lots?) of code while using the builtin libs vs importing a new lib.
- Use type hints for all functions and methods, including return types.
- Test any functionality you create with new pytest unit tests. Remember to run them with uv.
- I prefer not to use mocking when writing tests -- ideally you should test the actual functionality.
- Don't use superflous comments. Use comments to explain why something is done (if not obvious), not what is done.


## Build

The application is managed using `uv`, the list of packages is included in `pyproject.toml`. The development libraries can be installed with `uv sync --extra dev`.

