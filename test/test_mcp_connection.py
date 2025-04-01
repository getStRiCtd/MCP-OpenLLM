import asyncio

from src.mcp_classes.mcp_client import mcp_client

async def test_connection():
    async with mcp_client as client:
        tools = client.get_available_tools()
        return tools

if __name__ == "__main__":
    res = asyncio.run(test_connection())
    print(res)

