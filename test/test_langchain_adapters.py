import asyncio

from langgraph.checkpoint.memory import MemorySaver
from mcp import StdioServerParameters

from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools

from src.llm.llm_ import HuggingfaceModel
from src.mcp_classes.mcp_client import MCPClient

model = HuggingfaceModel()

server_params = StdioServerParameters(
    command="sudo",
    args=[
        "docker",
        "run",
        "--rm",  # Remove container after exit
        "-i",  # Interactive mode
        "-v",  # Mount volume
        "/media/ntfs_disk/Pycharn/mcp_local_test/mcp-test:/mcp",  # Map local volume to container path
        "mcp/sqlite",  # Use SQLite MCP image
        "--db-path",
        "/mcp/test.db",  # Database file path inside container
    ],
    env=None
)

def chat(thread_id: str, llm):
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        rq = input("\nHuman: ")
        print("User: ", rq)
        if rq == "":
            break
        resp = llm.invoke({"messages": [("user", rq)]}, config=config)
        print("Assistant: ", resp["messages"][-1].content)

async def test():
    async with MCPClient(server_params) as client:
        tools = await load_mcp_tools(client.session)

        agent = create_react_agent(model, tools, checkpointer=MemorySaver())
        chat("123", agent)

if __name__ == "__main__":
    asyncio.run(test())
