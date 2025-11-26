"""极简示例：实例化 LLMClient 并调用 invoke。"""

from __future__ import annotations

from client import LLMClient, LLMInvocationError


def main() -> None:
    client = LLMClient()  # 默认读取 pipeline/llm/config.llm.json
    try:
        result = client.invoke("测试一下自动回退机制是否正常？")
        print("LLM 回复：", result)
    except LLMInvocationError as exc:
        print("所有 provider 都失败了：")
        print(exc)


if __name__ == "__main__":
    main()
