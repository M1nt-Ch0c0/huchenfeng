# LLM 流程模块

这个模块允许你只调用一个函数就触发 LLM 推理，并会按照 `config.llm.json` 中的顺序，在 provider、model、API key 之间自动回退重试。

## 1. 配置 Provider

1. 将 `pipeline/llm/config.sample.json` 复制为 `pipeline/llm/config.llm.json`。
2. 把示例中的占位 API key 换成你自己的，可以直接写明文，或用 `$ENV:MY_VAR` 语法引用环境变量。
3. 按优先级排列 provider / model / key，客户端会严格按该顺序依次尝试，直到某次调用成功。

每个 provider 节点包含以下字段：

- `name`：日志/错误显示用标签。
- `driver`：请求/响应协议类型（`openai_chat` 与 `deepseek_chat` 共用 OpenAI Chat schema）。
- `base_url`：聊天补全接口地址。
- `timeout`：单次请求超时时间（秒，默认 60）。
- `headers`：可选的固定请求头。
- `default_params`：应用在此 provider 下所有模型的公共 JSON 参数（例如 `temperature`、`max_tokens`）。
- `response_path`：可选，用于指定如何从响应里提取 assistant 内容（默认 `["choices", 0, "message", "content"]`）。
- `json_format_field`：可选，若填写则表示该 provider 支持/要求一个布尔类型的 `json_format` 字段，调用时传入 `json_format=True/False` 会写入此字段。
- `models`：模型数组，分别包含 `name`、`keys`，以及可选的 `params`、`base_url`、`response_path`、`jsonable` 等字段。其中 `jsonable`（布尔值）表示该模型是否支持 JSON 模式；只有在 provider 设置了 `json_format_field` 且模型 `jsonable=true` 的情况下，`json_format` 参数才会写入请求体。

API key 写法：

- 支持直接写字符串、`$ENV:VARIABLE` 快捷语法，或 `{"env": "VARIABLE"}` 对象。
- 模块会在日志/错误中隐藏 key 的真实内容，并按顺序依次尝试。

## 2. 在代码中调用

```python
from pipeline.llm import invoke_llm

answer = invoke_llm(
    "请给我一句激励的话",
    system_prompt="你是户晨风知识库助手",
    temperature=0.4,          # 可选的参数覆盖
    max_tokens=500,
)
print(answer)
```

- `invoke_llm` 默认读取 `config.llm.json`；也可通过 `config_path` 或环境变量 `LLM_CONFIG_PATH` 指向其它配置。
- 若需提供完整对话历史，可传入 `messages=[...]` 代替单一 `prompt`。
- 需要强制 JSON 输出时传入 `json_format=True`；只有当 provider 配置了 `json_format_field` 且当前模型 `jsonable=true` 时，payload 才会包含该字段，其余情况自动忽略。
- 设置 `reload_config=True` 能在每次调用前强制重新加载配置文件。
- 高级用法可直接实例化 `pipeline.llm.client.LLMClient`，便于做依赖注入、自定义日志等。

当某次调用失败时，客户端会自动尝试下一组 key/model/provider，并记录失败原因；只有所有配置都失败时才会抛出 `LLMInvocationError`，其消息中包含所有尝试的摘要，方便排查问题。
