import json
import re
import ast
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

@register("model_mcp_bridge", "Dt8333", "一个用于为不支持tool_use的模型提供一个调用MCP的途径的插件", "1.0.2")
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
                request.system_prompt += "\n\nWhen using a tool, you can choose one of the following formats:\n\n**Format 1 - JSON (preferred):**\n{\n  \"tool\": \"tool_name\",\n  \"parameters\": {\n    \"param1\": \"value1\",\n    \"param2\": \"value2\"\n  },\n  \"call_id\": \"call_24CHaracterLOngSTRPlains\"\n}\n\n**Format 2 - tool_code (alternative):**\n```tool_code\nprint(default_api.tool_name(param1=\"value1\", param2=\"value2\"))\n```\n\nImportant rules:\n1. When using a tool: Output ONLY the tool call in one of the above formats, nothing else\n2. For JSON format: No markdown, no code blocks, no surrounding text\n3. For tool_code format: Use proper Python function call syntax with print() wrapper and default_api prefix\n4. Call exactly ONE tool per response\n5. When not using tools: Respond normally to the user's request\n\nExamples:\n- JSON: {\"tool\": \"search_web\", \"parameters\": {\"query\": \"weather today\"}, \"call_id\": \"call_123\"}\n- tool_code: ```tool_code\nprint(default_api.search_web(query=\"weather today\"))\n```"

    @filter.on_llm_response()
    async def onLlmResponse(self, event: AstrMessageEvent, response: LLMResponse) -> None:
        """这是一个在 LLM 响应时触发的事件"""
        if response.result_chain is not None:
            resp=response.result_chain.get_plain_text()

            # 首先尝试解析 Gemini 的 tool_code 格式
            tool_call_info = _parse_gemini_tool_call(resp)
            if tool_call_info:
                logger.debug("resp: "+resp)
                logger.info("Model calling tool by ModelMcpBridge (Gemini format), Converting.")
                response.tools_call_name = [tool_call_info["tool_name"]]
                response.tools_call_args = [tool_call_info["parameters"]]
                response.tools_call_ids = [tool_call_info["call_id"]]
                response.result_chain = None
                response.completion_text = ""

                """
                PART OF MONKEY PATCH
                调用存储的Agent实例的_transition_state方法，将状态设置为RUNNING，以继续处理工具调用
                """
                AGENT_STORAGE.get_agent(id(response))._transition_state(AgentState.RUNNING)
            else:
                # 如果不是 Gemini 格式，尝试解析 JSON 格式
                for resp_json in extract_json(resp):
                    if "tool" in resp_json and "parameters" in resp_json:
                        logger.debug("resp: "+resp)
                        logger.info("Model calling tool by ModelMcpBridge (JSON format), Converting.")
                        response.tools_call_name = [resp_json["tool"]]
                        response.tools_call_args = [resp_json["parameters"]]
                        response.tools_call_ids = [resp_json["call_id"]]
                        response.result_chain = None
                        response.completion_text = ""

                        """
                        PART OF MONKEY PATCH
                        调用存储的Agent实例的_transition_state方法，将状态设置为RUNNING，以继续处理工具调用
                        """
                        AGENT_STORAGE.get_agent(id(response))._transition_state(AgentState.RUNNING)

        """
        PART OF MONKEY PATCH
        从AgentStorage中移除已处理的Agent实例
        """
        AGENT_STORAGE.remove_agent(id(response))

    async def is_model_tool_use_support(self, provider: Provider, model: str) -> bool:
        """检查模型是否支持 tool_use 的示例函数"""
        if model is None:
            model=""
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

def _parse_gemini_tool_call(response_text: str) -> dict | None:
    """解析 Gemini 的 tool_code 格式，支持前后缀文本"""

    # 查找 ```tool_code 块，忽略前后的其他文本
    tool_code_pattern = r'```tool_code\s*\n(.*?)\n```'
    match = re.search(tool_code_pattern, response_text, re.DOTALL)

    if not match:
        return None

    tool_code = match.group(1).strip()

    # 记录完整的响应文本用于调试
    logger.debug(f"Found tool_code block: {tool_code}")
    logger.debug(f"Full response text: {response_text}")

    try:
        # 首先检查是否有外层的 print() 包装
        print_pattern = r'print\s*\(\s*(.*)\s*\)\s*$'
        print_match = re.match(print_pattern, tool_code, re.DOTALL)

        if print_match:
            # 如果有 print 包装，提取内部的函数调用
            inner_code = print_match.group(1).strip()
        else:
            # 如果没有 print 包装，直接使用原始代码
            inner_code = tool_code

        # 解析函数调用格式，支持带点的函数名，例如: default_api.fetch_html(url = "...")
        func_pattern = r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*\((.*)\)'
        func_match = re.match(func_pattern, inner_code, re.DOTALL)

        if not func_match:
            logger.debug(f"Failed to match function pattern in: {inner_code}")
            return None

        full_func_name = func_match.group(1)
        args_str = func_match.group(2).strip()

        # 提取实际的工具名（去掉前缀，如 default_api.）
        if '.' in full_func_name:
            tool_name = full_func_name.split('.')[-1]  # 取最后一部分作为工具名
        else:
            tool_name = full_func_name

        logger.debug(f"Extracted tool name: {tool_name} from full name: {full_func_name}")

        # 解析参数
        parameters = {}
        if args_str:
            # 处理参数字符串，支持各种类型的参数
            try:
                # 构建一个可以被 ast.literal_eval 解析的字符串
                # 将函数调用转换为字典格式
                eval_str = f"dict({args_str})"
                parameters = ast.literal_eval(eval_str)
            except (ValueError, SyntaxError):
                # 如果 ast.literal_eval 失败，尝试手动解析
                parameters = _parse_function_args(args_str)

        return {
            "tool_name": tool_name,
            "parameters": parameters,
            "call_id": f"call_{hash(tool_code) % (10**20):020d}"  # 生成一个20位的call_id
        }

    except Exception as e:
        logger.error(f"Failed to parse Gemini tool call: {e}")
        return None

def _parse_function_args(args_str: str) -> dict:
    """手动解析函数参数字符串，支持带空格的参数格式"""
    parameters = {}

    # 改进的参数匹配模式，支持更多空格和复杂的字符串值
    # 匹配格式: param = "value", param="value", param = 123, param=True 等
    param_pattern = r'(\w+)\s*=\s*(["\']([^"\']*)["\']|[^,\s)]+)'

    for match in re.finditer(param_pattern, args_str):
        param_name = match.group(1)
        param_value_full = match.group(2).strip()
        param_value_inner = match.group(3)

        # 如果是字符串（有引号）
        if param_value_inner is not None:
            parameters[param_name] = param_value_inner
        else:
            # 尝试转换为适当的类型
            param_value = param_value_full
            if param_value.lower() == 'true':
                parameters[param_name] = True
            elif param_value.lower() == 'false':
                parameters[param_name] = False
            elif param_value.lower() == 'null' or param_value.lower() == 'none':
                parameters[param_name] = None
            elif param_value.isdigit():
                parameters[param_name] = int(param_value)
            elif re.match(r'^\d+\.\d+$', param_value):
                parameters[param_name] = float(param_value)
            else:
                parameters[param_name] = param_value

    logger.debug(f"Parsed parameters: {parameters} from args_str: {args_str}")
    return parameters


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
            AGENT_STORAGE.set_agent(id(self.final_llm_resp), self)
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
