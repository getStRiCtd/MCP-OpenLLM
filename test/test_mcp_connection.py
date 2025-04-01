import asyncio

from mcp import StdioServerParameters

from src.mcp_classes.mcp_client import MCPClient


async def test_connection(server_params):
    async with MCPClient(server_params) as client:
        tools = await client.get_available_tools()
        return tools

if __name__ == "__main__":
    server_params = StdioServerParameters(
        command="sudo",
        args=[
            "docker",
            "run",
            "--rm",  # Remove container after exit
            "-i",  # Interactive mode
            "-v",  # Mount volume
            "mcp-test:/mcp",  # Map local volume to container path
            "mcp/sqlite",  # Use SQLite MCP image
            "--db-path",
            "/mcp/test.db",  # Database file path inside container
        ],
        env=None
    )

    res = asyncio.run(test_connection(server_params))
    print(res)

