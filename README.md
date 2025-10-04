# Astrobot Plugin Model MCP Bridge

## 简介
此插件为不原生支持 tool_use 或 tool_result 的模型提供兼容桥接。它通过在请求阶段向模型注入工具列表（JSON 格式）并在响应阶段解析模型返回的 JSON 来实现：当模型以指定的 JSON 格式请求调用工具时，插件会将该请求转换为 AstrBot 的内部工具调用；工具执行完成后，插件会把结果以文本形式回写到后续的 prompt 中，使模型能够继续基于工具输出生成后续响应，从而形成完整的调用-执行-回传闭环。

## 功能概述
- 探测模型是否支持 tool_use 与 tool_result。
- 在模型不支持 tool_use 时，将可用工具序列化为 JSON 并注入 system_prompt，引导模型以 JSON 格式发起工具调用。
- 在模型响应中解析 JSON 工具调用并转换为 AstrBot 的工具调用（填充 tools_call_* 字段），触发工具执行。
- 在模型不支持 tool_result 时，将工具执行结果即时追加到下一次请求的 prompt 中，确保结果能被模型继续使用。

## 工作原理（简要）
1. 检测：通过模拟工具调用与注入工具结果的方式，判断目标模型是否支持 tool_use 与 tool_result。
2. 请求时：若检测到模型不支持 tool_use，插件会把完整工具集（OpenAI 风格的 schema）序列化为 JSON，并将使用说明追加到 system_prompt，告诉模型如何以严格的 JSON 格式返回工具调用信息。
3. 响应时：插件解析模型输出中的 JSON（寻找包含 "tool" 与 "parameters" 的对象），并将其转换为 AstrBot 的工具调用数据结构，随后触发工具执行。
4. 结果回写：工具执行结果会被捕获并按文本形式追加到下一次 prompt 中（当模型不支持 tool_result 时），以便模型能基于该结果继续推理。

## 注意事项与风险
- 本插件部分实现依赖对 AstrBot 内部流程的 monkey patch（例如替换 ToolLoopAgentRunner 的状态转换和 FunctionToolExecutor 的执行逻辑），这是一种权衡实现方式，可能在 AstrBot 的内部实现变动时引发兼容性问题。
- 插件在捕获/注入数据时需谨慎，避免泄露敏感信息或造成 prompt 注入攻击风险；在生产环境中请对 system_prompt 的追加内容进行必要过滤与审查。
- 本插件在 AstrBot 4.0.0 上开发和测试，低版本 AstrBot 可能无法正常工作，需要适配。

## 许可证与仓库
该插件的源代码托管于: https://github.com/Dt8333/astrbot_plugin_model_mcp_bridge

