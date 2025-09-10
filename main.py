import json
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, FunctionTool, ToolSet
from astrbot.api.provider import ProviderRequest, LLMResponse, Provider
from dataclasses import dataclass, field

from astrbot.core.provider.func_tool_manager import FunctionToolManager
from astrbot.core.agent.runners.base import AgentState
from astrbot.core.agent.runners import tool_loop_agent_runner
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner

"""
实现思路：
在 onLlmRequest 中检测模型是否支持 tool_use，如果不支持，则将工具集序列化为 JSON，并将其添加到 system_prompt 中，指导模型如何调用工具。
在 onLlmResponse 中解析模型的响应，如果检测到模型尝试调用工具（通过解析 JSON 格式的响应），则将其转换为 AstrBot 的工具调用格式。
如果模型不支持 tool_use，则通过补丁的方式修改 ToolLoopAgentRunner 的 _transition_state 方法，以便在工具调用后正确地继续处理。
需要注意的是，这种方法依赖于模型能够理解并遵循 system_prompt 中的指示，因此效果可能会因模型而异。

改进方向：
等待合适的接口，能够在AGENT-RUNNING->AGENT-DONE或所有状态变化前hook，从而不必在onLlmResponse中调用_transition_state
"""

class AgentStorage:
    """
    #PART OF MONKEY PATCH
    用于临时储存Agent实例，以便在onLlmResponse中调用其_transition_state方法
    """
    def __init__(self):
        self.agent={}

    def set_agent(self, id, agent):
        self.agent[id]=agent

    def get_agent(self, id):
        return self.agent[id]

    def remove_agent(self, id):
        if id in self.agent:
            del self.agent[id]

AGENT_STORAGE=AgentStorage()

@register("model_mcp_bridge", "Dt8333", "一个用于为不支持tool_use的模型提供一个调用MCP的途径的插件", "1.0.1")
class ModelMcpBridge(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        self.ModelSupportToolUse={} # 用于存储模型是否支持 tool_use 的字典，key 为模型名，value 为布尔值

        """
        PART OF MONKEY PATCH
        用自定义的_transition_state方法替换ToolLoopAgentRunner的_transition_state方法
        以便在状态转换时捕获Agent实例
        """
        self._transition_state_backup=ToolLoopAgentRunner._transition_state
        ToolLoopAgentRunner._transition_state=_patched_transition_state

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
        provider=self.context.get_using_provider(event.unified_msg_origin)
        if not await self.is_model_tool_use_support(provider, request.model):
            logger.info("Model does not support tool_use, ModelMcpBridge Hooking.")
            toolSet: FunctionToolManager | ToolSet | None = request.func_tool
            if isinstance(toolSet, FunctionToolManager):
                request.func_tool = toolSet.get_full_tool_set()
                toolSet = request.func_tool

            """Serialize the tool set to JSON format"""
            jsonData = json.dumps(toolSet.openai_schema())

            if hasattr(request, 'system_prompt'):
                request.system_prompt += f"\n\nAvailable tools:\n{jsonData}\n\nUse tools when necessary."
                request.system_prompt += "\n\nWhen using a tool, respond with ONLY the following JSON format (no additional text, no markdown, no explanations):\n{\n  \"tool\": \"tool_name\",\n  \"parameters\": {\n    \"param1\": \"value1\",\n    \"param2\": \"value2\"\n  },\n  \"call_id\": \"call_24CHaracterLOngSTRPlains\"\n}\n\nImportant rules:\n1. When using a tool: Output ONLY the raw JSON, nothing else\n2. No markdown, no code blocks, no surrounding text of any kind\n3. Call exactly ONE tool per response\n4. When not using tools: Respond normally to the user's request"

    @filter.on_llm_response()
    async def onLlmResponse(self, event: AstrMessageEvent, response: LLMResponse) -> None:
        if response.result_chain is not None:
            """这是一个在 LLM 响应时触发的事件"""
            resp=response.result_chain.get_plain_text()
            for resp_json in extract_json(resp):
                if "tool" in resp_json and "parameters" in resp_json:
                    print("Model calling tool by ModelMcpBridge, Converting.")
                    response.tools_call_name = [resp_json["tool"]]
                    response.tools_call_args = [resp_json["parameters"]]
                    response.tools_call_ids = [resp_json["call_id"]]
                    response.result_chain = None
                    response.completion_text = ""

                    """
                    PART OF MONKEY PATCH
                    调用存储的Agent实例的_transition_state方法，将状态设置为RUNNING，以继续处理工具调用
                    """
                    AGENT_STORAGE.get_agent(response.raw_completion.id)._transition_state(AgentState.RUNNING)

        """
        PART OF MONKEY PATCH
        从AgentStorage中移除已处理的Agent实例
        """
        AGENT_STORAGE.remove_agent(response.raw_completion.id)

    async def is_model_tool_use_support(self, provider: Provider, model: str) -> bool:
        """检查模型是否支持 tool_use 的示例函数"""
        key=provider.meta().id+"_"+model
        if key not in self.ModelSupportToolUse:
            MockToolset=ToolSet([MockTool()])
            llm_resp = await provider.text_chat(
                prompt="What is the temperature of my CPU?",
                system_prompt="You are a helpful assistant. Use get_cpu_temperature tool to answer the question.",
                func_tool=MockToolset,
                model=model
            )

            if MockTool.name not in llm_resp.tools_call_name:
                self.ModelSupportToolUse[key]=False
            else:
                self.ModelSupportToolUse[key]=True

        return self.ModelSupportToolUse.get(key, False)

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""

        """
        PART OF MONKEY PATCH
        恢复ToolLoopAgentRunner的_transition_state方法，以使其能在插件卸载后正常工作
        """
        ToolLoopAgentRunner._transition_state=self._transition_state_backup

def RawJSONDecoder(index):
    class _RawJSONDecoder(json.JSONDecoder):
        end = None

        def decode(self, s, *_):
            data, self.__class__.end = self.raw_decode(s, index)
            return data
    return _RawJSONDecoder

"""
工具类
解析JSON字符串
"""
def extract_json(s, index=0):
    while (index := s.find('{', index)) != -1:
        try:
            yield json.loads(s, cls=(decoder := RawJSONDecoder(index)))
            index = decoder.end
        except json.JSONDecodeError:
            index += 1

def _patched_transition_state(self, new_state: AgentState) -> None:
    """
    PART OF MONKEY PATCH
    修改后的_transition_state方法
    用于捕获Agent实例
    """

    """转换 Agent 状态"""
    if self._state != new_state:
        if self._state == AgentState.RUNNING and new_state == AgentState.DONE:
            AGENT_STORAGE.set_agent(self.final_llm_resp.raw_completion.id,self)
        logger.debug(f"Agent state transition: {self._state} -> {new_state}")
        self._state = new_state

@dataclass
class MockTool(FunctionTool):
    name: str = "get_cpu_temperature"
    description: str = "A tool to get the current CPU temperature of the user's device."
    parameters: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Random input string to avoid caching"
            }
        },
        "required": ["input"]
    })

    async def run(self, input: str) -> str:
        return f"Mock tool received input: {input}"
