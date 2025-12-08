from transformers import AutoTokenizer
import os

model_path = "./merged_model_final"

print(f"🔧 正在修复 {model_path} 的对话模板...")

# 1. 加载本地的分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. Qwen2.5 的标准 ChatML 模板
# 这是一段 Jinja2 代码，告诉分词器怎么处理 system/user/assistant 消息
qwen_chat_template = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

# 3. 将模板赋值给分词器
tokenizer.chat_template = qwen_chat_template

# 4. 保存回本地 (覆盖旧的 tokenizer_config.json)
print("💾 正在保存修复后的配置...")
tokenizer.save_pretrained(model_path)

print("✅ 修复完成！现在的模型文件夹已经包含了正确的 chat_template。")
print("请再次运行 python test.py")