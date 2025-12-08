---
language:
  - zh
license: other
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen
  - lora
  - dialogue
  - chinese
---

# Model Card: HuChenFeng Qwen2.5-7B LoRA

## Model Details
- **Model type:** Conversational LLM fine-tuned to emulate the speaking style of the Chinese streamer 户晨风.
- **Base model:** [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
- **Adaptation method:** LoRA (r=16, alpha=16) with 4-bit NormalFloat quantization, trained via [Unsloth](https://github.com/unslothai/unsloth) + SFT.
- **Training hardware:** Single RTX 4090 (≈7 hours total fine-tuning time).
- **Intended maintainers:** HuChenFeng project authors (see [GitHub repo](https://github.com/tinymindkin/huchenfeng)).

## Model Sources
- **Repository:** https://github.com/tinymindkin/huchenfeng
- **Dataset location:** Provided alongside the repository under `dataset/` (ChatML JSON format).

## Uses
### 学术/研究（For study）
- 为研究者探索人设约束、安全提示或记忆模块如何影响高度风格化助手的提示工程提供实验场。
- 可作为覆盖 LoRA、QLoRA 以及中文对话评测课程或教程的基线模型检查点。

### 娱乐/创意（For fun）
- 适合角色聊天机器人、直播搭档模拟或需要夸张长篇叙述的互动小说应用。
- 支持“户晨风机器人”等社群挑战或角色扮演活动，方便粉丝改编台词与即兴发挥。


## Dataset
- **Size:** 80,137 dialogue pairs; average question length 18.3 Chinese characters, average answer length 342.7 characters; ≈2.8B tokens (Qwen tokenizer).
- **Source:** 2023-2024 livestream transcripts (>200 words per utterance) collected from 户晨风’s streams.
- **Processing pipeline:**
  1. Whisper Large-v3 for speech-to-text transcription (~2M words raw).
  2. Gemini-2.5-Flash cleaning (remove short segments, repeated content, background chatter, comment reading, obvious ASR errors) with cost ≈ \$42.
  3. Gemini-2.5-Flash-Lite prompt-driven question generation (3–5 prompts per segment) with cost ≈ \$18.
  4. Post-filtering for length, duplication, sentiment; final 80K ChatML-format pairs released publicly, with 12K high-confidence entries used for fine-tuning.
- **Format:** ChatML-style JSON with alternating `user` and `assistant` messages.

## Training Procedure
### Hyperparameters
```yaml
learning_rate: 2e-4
batch_size: 4 (physical) x 4 (grad accumulation) = 16 effective
epochs: 3
optimizer: AdamW-8bit
warmup_steps: 10
max_seq_length: 2048
total_steps: ~2250
```
- LoRA: r=16, alpha=16, dropout default (0.1). Adapter layers applied to attention projections.
- Quantization: 4-bit NormalFloat (bitsandbytes) to reduce VRAM consumption.

### Data Sampling
- Length-based stratified sampling to preserve mix of short and long responses.
- Deduplication threshold 60% to avoid near-identical segments.
- Manual spot-checking to keep persona-consistent segments only.

### Training Infrastructure
- Single RTX 4090 (24 GB VRAM) using Unsloth accelerated fine-tuning.
- Gradient checkpointing enabled; logging via TRL.


## How to Use
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "YOUR MODEL PATH"
# 1. load model
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype="auto", 
    trust_remote_code=True
)

# 2. load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "你怎么看待大专毕业的职业选择？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
 prompt = "请做一下自我介绍。"
    messages = [
        {"role": "system", "content": '''你是户晨风，回答时必须遵循以下规则：

【核心原则】
1. 先直接回答问题，再展开说明
2. 回答必须紧扣用户的问题
3. 如果不确定，说"这个我不太了解"
'''},
{"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)
for output in outputs:
print(f"🤖 回答: {tokenizer.decode(output, skip_special_tokens=True)}")
```


## Contact
GitHub: https://github.com/tinymindkin/huchenfeng
