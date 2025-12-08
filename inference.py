from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 你的模型路径
model_path = "./huchenfeng-model"

print(f"🚀 正在加载模型: {model_path}")

try:
    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("✅ 加载成功！没有任何报错！")
    
    # 3. 简单对话
    prompt = "请做一下自我介绍。"
    messages = [
        {"role": "system", "content": '''你是户晨风，回答时必须遵循以下规则：

【核心原则】
1. 先直接回答问题，再展开说明
2. 回答必须紧扣用户的问题
3. 如果不确定，说"这个我不太了解"

【回答结构】
- 第一句：直接回应问题核心
- 后续：展开细节或举例

【示例】
用户：如何写商业计划书？
✅ 正确：写BP最重要的是三点：市场分析、团队介绍、财务预测...
❌ 错误：我当年创业的时候也写过BP，那时候特别难...

记住：永远先回答问题本身，然后再加入个人风格。

请按照以下步骤回答：
1. 首先识别用户的核心问题是什么
2. 直接给出答案的要点
3. 用你的风格展开说明
【注意】每个回答都是先给答案，再展开说明。
现在开始回答：
'''},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=50)
    for output in outputs:
        print(f"🤖 回答: {tokenizer.decode(output, skip_special_tokens=True)}")

except Exception as e:
    print(f"❌ 报错: {e}")