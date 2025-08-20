import json
from unittest.mock import AsyncMock

import pytest

from zipcoil import Agent, tool


def test_agent_rejects_async_tools():
    @tool
    def sync_tool(x: int) -> int:
        return x

    @tool
    async def async_tool(x: int) -> int:
        return x

    mock_client = AsyncMock()

    with pytest.raises(ValueError, match=r"async_tool.*?an async function"):
        Agent(model="gpt-4", client=mock_client, tools=[sync_tool, async_tool])
