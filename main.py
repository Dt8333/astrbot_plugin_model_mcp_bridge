import json
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger,FunctionTool,ToolSet
from astrbot.api.provider import ProviderRequest, LLMResponse
from dataclasses import field

from astrbot.core.provider.func_tool_manager import FunctionToolManager
from astrbot.core.agent.runners.base import AgentState
from astrbot.core.agent.runners import tool_loop_agent_runner
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner

class AgentStorage:
    def __init__(self):
        self.agent=None

    def set_agent(self,agent):
        self.agent=agent

    def get_agent(self):
        return self.agent

AGENT_STORAGE=AgentStorage()

@register("model_mcp_bridge", "Dt8333", "一个用于为不支持tool_use的模型提供一个调用MCP的途径的插件", "1.0.0")
class ModelMcpBridge(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        self.ModelSupportToolUse={} # 用于存储模型是否支持 tool_use 的字典，key 为模型名，value 为布尔值
        self._transition_state_backup=ToolLoopAgentRunner._transition_state
        ToolLoopAgentRunner._transition_state=_patched_transition_state
        tool_loop_agent_runner.AGENT=None

    # 注册指令的装饰器。指令名为 mcpbridge。注册成功后，发送 `/mcpbridge` 就会触发这个指令，并回复 `你好, {user_name}!`
    @filter.command("mcpbridge")
    async def commandMcpBridge(self, event: AstrMessageEvent):
        """这是一个 mcpbridge 指令""" # 这是 handler 的描述，将会被解析方便用户了解插件内容。建议填写。
        user_name = event.get_sender_name()
        message_str = event.message_str # 用户发的纯文本消息字符串
        message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)
        yield event.plain_result(f"Hello, {user_name}, 你发了 {message_str}!") # 发送一条纯文本消息

    @filter.on_llm_request(priority=-10001)
    async def onLlmRequest(self, event: AstrMessageEvent, request: ProviderRequest) -> None:
        """这是一个在 LLM 请求时触发的事件"""
        if not await self.is_model_tool_use_support(request.model):
            print("Model does not support tool_use, ModelMcpBridge Hooking.")
            toolSet: FunctionToolManager | ToolSet | None = request.func_tool
            if isinstance(toolSet, FunctionToolManager):
                request.func_tool = tool_set.get_full_tool_set()
                tool_set = request.func_tool

            """Serialize the tool set to JSON format"""
            jsonData = json.dumps(toolSet.openai_schema())

            if hasattr(request, 'system_prompt'):
                request.system_prompt += f"\n\nAvailable tools:\n{jsonData}\n\nUse tools when necessary."
                request.system_prompt += "\n\nWhen using a tool, respond with ONLY the following JSON format (no additional text, no markdown, no explanations):\n{\n  \"tool\": \"tool_name\",\n  \"parameters\": {\n    \"param1\": \"value1\",\n    \"param2\": \"value2\"\n  },\n  \"call_id\": \"call_24CHaracterLOngSTRPlains\"\n}\n\nImportant rules:\n1. When using a tool: Output ONLY the raw JSON, nothing else\n2. No markdown, no code blocks, no surrounding text of any kind\n3. Call exactly ONE tool per response\n4. When not using tools: Respond normally to the user's request"

    @filter.on_llm_response()
    async def onLlmResponse(self, event: AstrMessageEvent, response: LLMResponse) -> None:
        if response.result_chain is None:
            return
        """这是一个在 LLM 响应时触发的事件"""
        resp=response.result_chain.get_plain_text()
        try:
            resp_json = json.loads(resp)
            if "tool" in resp_json and "parameters" in resp_json:
                print("Model calling tool by ModelMcpBridge, Converting.")
                response.tools_call_name = [resp_json["tool"]]
                response.tools_call_args = [resp_json["parameters"]]
                response.tools_call_ids = [resp_json["call_id"]]
                response.result_chain = None
                response.completion_text = ""
                AGENT_STORAGE.get_agent()._transition_state(AgentState.RUNNING)
        except json.JSONDecodeError:
            pass

    async def is_model_tool_use_support(self, model_name: str) -> bool:
        """检查模型是否支持 tool_use 的示例函数"""
        if model_name not in self.ModelSupportToolUse:
            provider=self.context.get_provider_by_id(model_name)
            if provider:
                MockToolset=ToolSet(FunctionTool)
                llm_resp = await provider.text_chat(
                    prompt="Call the mock tool with random string",
                    system_prompt="You are a helpful assistant that can use tools.",
                    func_tool=MockToolset
                )

                if MockTool.name not in llm_resp.tools_call_name:
                    self.ModelSupportToolUse[model_name]=False
                else:
                    self.ModelSupportToolUse[model_name]=True

        return self.ModelSupportToolUse.get(model_name, False)

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
        ToolLoopAgentRunner._transition_state=self._transition_state_backup


def _patched_transition_state(self, new_state: AgentState) -> None:
    """转换 Agent 状态"""
    if self._state != new_state:
        if self._state == AgentState.RUNNING and new_state == AgentState.DONE:
            AGENT_STORAGE.set_agent(self)
        logger.debug(f"Agent state transition: {self._state} -> {new_state}")
        self._state = new_state

class MockTool(FunctionTool):
    name: str = "mock_tool"
    description: str = "这是一个模拟工具，用于测试模型是否支持 tool_use 功能。"
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Random input string"
            }
        },
        "required": ["input"]
    })

    async def run(self, input: str) -> str:
        return f"Mock tool received input: {input}"
