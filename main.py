import json
import re
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, FunctionTool, ToolSet
from astrbot.api.provider import ProviderRequest, LLMResponse, Provider
from dataclasses import dataclass, field

from astrbot.core.provider.entities import (
    AssistantMessageSegment,
    ToolCallMessageSegment,
    ToolCallsResult,
)
from astrbot.core.provider.func_tool_manager import FunctionToolManager
from astrbot.core.agent.runners.base import AgentState
from astrbot.core.agent.runners import tool_loop_agent_runner
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
import random

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
        self.agent = {}

    def set_agent(self, id, agent):
        self.agent[id] = agent

    def get_agent(self, id):
        return self.agent[id]

    def remove_agent(self, id):
        if id in self.agent:
            del self.agent[id]


AGENT_STORAGE = AgentStorage()


@register(
    "model_mcp_bridge",
    "Dt8333",
    "一个用于为不支持tool_use的模型提供一个调用MCP的途径的插件",
    "1.1.0",
)
class ModelMcpBridge(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        self.ModelSupportToolUse = (
            {}
        )  # 用于存储模型是否支持 tool_use 的字典，key 为模型名，value 为布尔值
        self.ModelSupportToolResult = (
            {}
        )  # 用于存储模型是否支持 tool_result 的字典，key 为模型名，value 为布尔值



        """
        PART OF MONKEY PATCH
        用自定义的_transition_state方法替换ToolLoopAgentRunner的_transition_state方法
        以便在状态转换时捕获Agent实例
        """
        self._transition_state_backup = ToolLoopAgentRunner._transition_state
        ToolLoopAgentRunner._transition_state = _patched_transition_state

        """
        动态替换 MAIN_AGENT_HOOKS
        """
        from astrbot.core.pipeline.process_stage.method.llm_request import MAIN_AGENT_HOOKS
        self.original_main_agent_hooks = MAIN_AGENT_HOOKS

        import astrbot.core.pipeline.process_stage.method.llm_request as llm_request_module
        self.custom_hooks = CustomMainAgentHooks(self)
        llm_request_module.MAIN_AGENT_HOOKS = self.custom_hooks

        """
        PART OF MONKEY PATCH
        Monkey patch FunctionToolExecutor.execute 来捕获工具结果
        """
        from astrbot.core.pipeline.process_stage.method.llm_request import FunctionToolExecutor
        self._original_execute = FunctionToolExecutor.execute
        FunctionToolExecutor.execute = self._create_patched_execute()

    # 注册指令的装饰器。指令名为 mcpbridge。注册成功后，发送 `/mcpbridge` 就会触发这个指令，并回复 `你好, {user_name}!`
    @filter.command("mcpbridge")
    async def commandMcpBridge(self, event: AstrMessageEvent):
        """这是一个 mcpbridge 指令"""  # 这是 handler 的描述，将会被解析方便用户了解插件内容。建议填写。
        user_name = event.get_sender_name()
        message_str = event.message_str  # 用户发的纯文本消息字符串
        message_chain = (
            event.get_messages()
        )  # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)
        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider:
            yield event.plain_result("未找到当前使用的提供商")
            return
        model = provider.get_model() or ""
        support_tool_use = await self.is_model_tool_use_support(provider, model)
        support_tool_result = await self.is_model_tool_result_support(provider, model)
        logger.info(
            f"Provider: {provider.meta().id}, Model: {model}, Support tool_use: {support_tool_use}, Support tool_result: {support_tool_result}"
        )
        yield event.plain_result(
            f"Provider:{provider.meta().id}, Model:{model}, Support tool_use:{support_tool_use}, Support tool_result:{support_tool_result}"
        )  # 发送一条纯文本消息

    @filter.on_llm_request(priority=-10001)
    async def onLlmRequest(
        self, event: AstrMessageEvent, request: ProviderRequest
    ) -> None:
        """这是一个在 LLM 请求时触发的事件"""
        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not await self.is_model_tool_use_support(provider, request.model):
            logger.info("Model does not support tool_use, ModelMcpBridge Hooking.")
            toolSet: FunctionToolManager | ToolSet | None = request.func_tool
            if isinstance(toolSet, FunctionToolManager):
                request.func_tool = toolSet.get_full_tool_set()
                toolSet = request.func_tool

            """Serialize the tool set to JSON format"""
            jsonData = json.dumps(toolSet.openai_schema())

            if hasattr(request, "system_prompt"):
                request.system_prompt += (
                    f"\n\nAvailable tools:\n{jsonData}\n\nUse tools when necessary."
                )
                request.system_prompt += '\n\nWhen using a tool, use the following JSON format:\n\n{\n  "tool": "tool_name",\n  "parameters": {\n    "param1": "value1",\n    "param2": "value2"\n  },\n  "call_id": "call_24CHaracterLOngSTRPlains"\n}\n\nImportant rules:\n1. When using a tool: Output ONLY the tool call in JSON format, nothing else\n2. No markdown, no code blocks, no surrounding text\n3. Call exactly ONE tool per response\n4. When not using tools: Respond normally to the user\'s request\n\nExample: {"tool": "search_web", "parameters": {"query": "weather today"}, "call_id": "call_123"}'

    @filter.on_llm_response()
    async def onLlmResponse(
        self, event: AstrMessageEvent, response: LLMResponse
    ) -> None:
        """这是一个在 LLM 响应时触发的事件"""
        if response.result_chain is not None:
            resp = response.result_chain.get_plain_text()

            # 解析 JSON 格式的工具调用
            for resp_json in extract_json(resp):
                if "tool" in resp_json and "parameters" in resp_json:
                    logger.debug("resp: " + resp)
                    logger.debug(
                        "name: "
                        + resp_json["tool"]
                        + ", args: "
                        + str(resp_json["parameters"])
                    )
                    logger.info(
                        "Model calling tool by ModelMcpBridge (JSON format), Converting."
                    )
                    response.tools_call_name = [resp_json["tool"]]
                    response.tools_call_args = [resp_json["parameters"]]
                    response.tools_call_ids = [resp_json["call_id"]]
                    response.result_chain = None
                    response.completion_text = ""

                    """
                    PART OF MONKEY PATCH
                    调用存储的Agent实例的_transition_state方法，将状态设置为RUNNING，以继续处理工具调用
                    """
                    AGENT_STORAGE.get_agent(id(response))._transition_state(
                        AgentState.RUNNING
                    )
                    break  # 只处理第一个有效的工具调用

        """
        PART OF MONKEY PATCH
        从AgentStorage中移除已处理的Agent实例
        """
        AGENT_STORAGE.remove_agent(id(response))

    async def is_model_tool_use_support(self, provider: Provider, model: str) -> bool:
        """检查模型是否支持 tool_use 的示例函数"""
        if model is None:
            model = ""
        key = provider.meta().id + "_" + model
        if key not in self.ModelSupportToolUse:
            MockToolset = ToolSet([MockTool()])
            llm_resp = await provider.text_chat(
                prompt="What is the temperature of my CPU?",
                system_prompt="You are a helpful assistant. Use get_cpu_temperature tool to answer the question.",
                func_tool=MockToolset,
                model=model,
            )

            if MockTool.name not in llm_resp.tools_call_name:
                self.ModelSupportToolUse[key] = False
            else:
                self.ModelSupportToolUse[key] = True

        return self.ModelSupportToolUse.get(key, False)

    async def is_model_tool_result_support(
        self, provider: Provider, model: str
    ) -> bool:
        """检查模型是否支持 tool_result 的示例函数"""
        if model is None:
            model = ""
        key = provider.meta().id + "_" + model
        if key not in self.ModelSupportToolResult:
            MockToolset = ToolSet([MockTool()])
            tool_call_info = AssistantMessageSegment(
                content="",
                tool_calls=[
                    {
                        "id": "call_24CHaracterLOngSTRPlains",
                        "function": {
                            "name": MockTool.name,
                            "arguments": json.dumps(
                                {
                                    "input": str(
                                        random.randint(100000, 999999)
                                    )  # 随机输入，避免缓存
                                }
                            ),
                        },
                        "type": "function",
                    }
                ],
                role="assistant",
            )
            tool_call_result = ToolCallsResult(
                tool_calls_info=tool_call_info,
                tool_calls_result=[
                    ToolCallMessageSegment(
                        role="tool",
                        tool_call_id=tool_call_info.tool_calls[0]["id"],
                        content="cpu temperature: 55",
                    )
                ],
            )
            llm_resp = await provider.text_chat(
                prompt="What is the temperature of my CPU?",
                system_prompt="You are a helpful assistant. Use get_cpu_temperature tool to answer the question.",
                func_tool=MockToolset,
                model=model,
                tool_calls_result=tool_call_result,
            )

            if llm_resp.completion_text.find("55") == -1:
                self.ModelSupportToolResult[key] = False
            else:
                self.ModelSupportToolResult[key] = True

        return self.ModelSupportToolResult.get(key, False)

    def _create_patched_execute(self):
        """创建 patched 的 FunctionToolExecutor.execute 方法"""
        original_execute = self._original_execute
        custom_hooks = self.custom_hooks

        @classmethod
        async def patched_execute(cls, tool, run_context, **tool_args):
            """执行函数调用的 patched 版本，会捕获工具结果"""
            from mcp.types import CallToolResult

            # 调用原始的 execute 方法
            async for result in original_execute(tool, run_context, **tool_args):
                if isinstance(result, CallToolResult):
                    # ✅ 关键：在工具结果生成时立即存储
                    if result.content:
                        from mcp.types import TextContent, ImageContent, EmbeddedResource, TextResourceContents, BlobResourceContents

                        content = result.content[0]
                        if isinstance(content, TextContent):
                            custom_hooks.current_tool_results[tool.name] = content.text
                        elif isinstance(content, ImageContent):
                            custom_hooks.current_tool_results[tool.name] = "[Image returned]"
                        elif isinstance(content, EmbeddedResource):
                            resource = content.resource
                            if isinstance(resource, TextResourceContents):
                                custom_hooks.current_tool_results[tool.name] = resource.text
                            elif isinstance(resource, BlobResourceContents) and resource.mimeType and resource.mimeType.startswith("image/"):
                                custom_hooks.current_tool_results[tool.name] = "[Image resource returned]"
                            else:
                                custom_hooks.current_tool_results[tool.name] = "返回的数据类型不受支持"
                        else:
                            custom_hooks.current_tool_results[tool.name] = str(content)

                yield result

        return patched_execute

    def _convert_single_tool_result_to_text(self, tool_name, tool_args, tool_result) -> str:
        """将单个工具调用结果转换为文本格式"""
        try:
            # 提取工具结果内容
            if hasattr(tool_result, 'content') and tool_result.content:
                from mcp.types import TextContent, ImageContent
                if isinstance(tool_result.content[0], TextContent):
                    result_content = tool_result.content[0].text
                elif isinstance(tool_result.content[0], ImageContent):
                    result_content = "[Image returned]"
                else:
                    result_content = str(tool_result.content[0])
            else:
                result_content = "No content returned"

            return (
                f"Tool Call: {tool_name}\n"
                f"Arguments: {json.dumps(tool_args) if tool_args else '{}'}\n"
                f"Result: {result_content}\n"
                f"---"
            )
        except Exception as e:
            logger.error(f"Error converting tool result to text: {e}")
            return f"Tool Call: {tool_name}\nResult: Error processing result\n---"

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""

        """
        PART OF MONKEY PATCH
        恢复ToolLoopAgentRunner的_transition_state方法，以使其能在插件卸载后正常工作
        """
        ToolLoopAgentRunner._transition_state = self._transition_state_backup

        """
        PART OF MONKEY PATCH
        恢复FunctionToolExecutor.execute方法
        """
        from astrbot.core.pipeline.process_stage.method.llm_request import FunctionToolExecutor
        FunctionToolExecutor.execute = self._original_execute

        """
        恢复原始的 MAIN_AGENT_HOOKS
        """
        import astrbot.core.pipeline.process_stage.method.llm_request as llm_request_module
        llm_request_module.MAIN_AGENT_HOOKS = self.original_main_agent_hooks





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
    while (index := s.find("{", index)) != -1:
        try:
            yield json.loads(s, cls=(decoder := RawJSONDecoder(index)))
            index = decoder.end
        except json.JSONDecodeError:
            index += 1


class CustomMainAgentHooks:
    """自定义的 MainAgentHooks，用于处理工具结果转换"""

    def __init__(self, bridge_plugin):
        self.bridge_plugin = bridge_plugin
        self.current_tool_results = {}  # key: (tool_name, args_hash), value: content

    async def on_agent_begin(self, run_context):
        """Agent 开始时触发"""
        pass

    async def on_tool_start(self, run_context, tool, tool_args):
        """工具开始执行时触发"""
        # 清空之前的结果
        key = (tool.name, hash(str(tool_args)))
        self.current_tool_results.pop(key, None)

    async def on_tool_end(self, run_context, tool, tool_args, tool_result):
        """工具执行完成时触发，立即将工具结果追加到 prompt 中"""
        try:
            provider = run_context.context.provider
            curr_req = run_context.context.curr_provider_request
            model = curr_req.model

            # 检查模型是否支持结构化工具调用结果
            if not await self.bridge_plugin.is_model_tool_result_support(provider, model):
                # 从存储中获取工具结果
                tool_result_content = self.current_tool_results.get(tool.name)

                if tool_result_content:
                    # 构造工具结果文本
                    tool_text = (
                        f"Tool Call: {tool.name}\n"
                        f"Arguments: {json.dumps(tool_args) if tool_args else '{}'}\n"
                        f"Result: {tool_result_content}\n"
                        f"---"
                    )

                    # 立即追加到当前请求的 prompt 中，为下次 LLM 调用做准备
                    if curr_req.prompt:
                        curr_req.prompt += f"\n\n{tool_text}"
                    else:
                        curr_req.prompt = tool_text

                    # 清空结构化的工具调用结果，避免重复处理
                    curr_req.tool_calls_result = None

                    logger.debug(f"Appended tool result to prompt: {tool.name}")
                else:
                    logger.warning(f"No tool result found for tool: {tool.name}")

            # ✅ 总是清理存储的结果，避免内存泄漏
            # 不论模型是否支持工具调用结果，都要清理存储
            self.current_tool_results.pop(tool.name, None)

        except Exception as e:
            logger.error(f"Error in on_tool_end hook: {e}")

    async def on_agent_done(self, run_context, llm_response):
        """Agent 完成时触发，只负责触发 OnLLMResponseEvent"""
        try:
            # 执行原有的事件钩子逻辑，让 onLlmResponse 检测并插入工具调用
            from astrbot.core.star.star_handler import EventType
            from astrbot.core.pipeline.context import call_event_hook
            await call_event_hook(
                run_context.event, EventType.OnLLMResponseEvent, llm_response
            )
            # 工具结果已在 on_tool_end 中立即处理并追加到 prompt
        except Exception as e:
            logger.error(f"Error in on_agent_done hook: {e}")


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
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Random input string to avoid caching",
                }
            },
            "required": ["input"],
        }
    )

    async def run(self, input: str) -> str:
        random.seed(hash(input))
        cpu_temperature = random.randint(30, 80)
        return f"cpu temperature: {cpu_temperature}"
