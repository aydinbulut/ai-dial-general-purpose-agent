import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import GeneralPurposeAgent  
from task.prompts import SYSTEM_PROMPT
from task.tools.base import BaseTool
from task.tools.deployment.image_generation_tool import ImageGenerationTool
from task.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.tools.py_interpreter.python_code_interpreter_tool import PythonCodeInterpreterTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.tools.rag.document_cache import DocumentCache
from task.tools.rag.rag_tool import RagTool

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
# DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'claude-sonnet-3-7')


class GeneralPurposeAgentApplication(ChatCompletion):

    def __init__(self):
        self.tools: list[BaseTool] = []

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        #TODO:
        # 1. Create list of BaseTool
        mcp_tools: list[BaseTool] = []
        # 2. Create MCPClient
        mcp_client = MCPClient(mcp_server_url=url)
        # 3. Get tools, iterate through them and add them to created list as MCPTool where the client will be created
        #    MCPClient and mcp_tool_model will be the tool itself (see what `mcp_client.get_tools` returns).
        for mcp_tool_model in await mcp_client.get_tools():
            mcp_tools.append(MCPTool(client=mcp_client, mcp_tool_model=mcp_tool_model))
        # 4. Return created tool list
        return mcp_tools

    async def _create_tools(self) -> list[BaseTool]:
        #TODO:
        # 1. Create list of BaseTool
        # ---
        # At the beginning this list can be empty. We will add here tools after they will be implemented
        # ---
        tools: list[BaseTool] = []
        # 2. Add ImageGenerationTool with DIAL_ENDPOINT
        tools.append(ImageGenerationTool(endpoint=DIAL_ENDPOINT))
        # 3. Add FileContentExtractionTool with DIAL_ENDPOINT
        tools.append(FileContentExtractionTool(endpoint=DIAL_ENDPOINT))
        # 4. Add RagTool with DIAL_ENDPOINT, DEPLOYMENT_NAME, and create DocumentCache (it has static method `create`)
        document_cache = DocumentCache.create()
        tools.append(RagTool(
            endpoint=DIAL_ENDPOINT,
            deployment_name=DEPLOYMENT_NAME,
            document_cache=document_cache,
        ))
        # 5. Add PythonCodeInterpreterTool with DIAL_ENDPOINT, `http://localhost:8050/mcp` mcp_url, tool_name is
        #    `execute_code`, more detailed about tools see in repository https://github.com/khshanovskyi/mcp-python-code-interpreter
        # tools.append(PythonCodeInterpreterTool(
        #     dial_endpoint=DIAL_ENDPOINT,
        #     mcp_url="http://localhost:8050/mcp",
        #     tool_name="execute_code",
        # ))
        # 6. Extend tools with MCP tools from `http://localhost:8051/mcp` (use method `_get_mcp_tools`)
        mcp_client = await MCPClient.create(mcp_server_url="http://localhost:8051/mcp")
        mcp_tools = mcp_client.get_tools()
        for mcp_tool_model in await mcp_tools:
            tools.append(MCPTool(client=mcp_client, mcp_tool_model=mcp_tool_model))
        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        #TODO:
        # 1. If `self.tools` are absent then call `_create_tools` method and assign to the `self.tools`
        if not self.tools:
            self.tools = await self._create_tools()
        # 2. Create `choice` (`with response.create_single_choice() as choice:`) and:
        #   - Create GeneralPurposeAgent with:
        #       - endpoint=DIAL_ENDPOINT
        #       - system_prompt=SYSTEM_PROMPT
        #       - tools=self.tools
        #   - call `handle_request` on created agent with:
        #       - choice=choice
        #       - deployment_name=DEPLOYMENT_NAME
        #       - request=request
        #       - response=response
        with response.create_single_choice() as choice:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools,
            )
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response,
            )

#TODO:
# 1. Create DIALApp
CREATED_DIAL_APP = DIALApp()
# 2. Create GeneralPurposeAgentApplication
agent_app = GeneralPurposeAgentApplication()
# 3. Add to created DIALApp chat_completion with:
#       - deployment_name="general-purpose-agent"
#       - impl=agent_app
CREATED_DIAL_APP.add_chat_completion(
    deployment_name="general-purpose-agent",
    impl=agent_app,
)
# 4. Run it with uvicorn: `uvicorn.run({CREATED_DIAL_APP}, port=5030, host="0.0.0.0")`
uvicorn.run(CREATED_DIAL_APP, port=5030, host="0.0.0.0")
