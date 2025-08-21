from unittest.mock import Mock

import pytest

from zipcoil import Agent, tool


def test_agent_rejects_async_tools():
    @tool
    def sync_tool(x: int) -> int:
        return x

    @tool
    async def async_tool(x: int) -> int:
        return x

    mock_client = Mock()

    with pytest.raises(ValueError, match=r"async_tool.*?an async function"):
        Agent(model="gpt-4", client=mock_client, tools=[sync_tool, async_tool])
