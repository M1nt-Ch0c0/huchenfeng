from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_PATH = os.path.join(ROOT_DIR, "sft.jsonl")

# ============ 配置参数 ============
max_seq_length = 2048  # 最大序列长度，根据对话长度调整，越长显存占用越大
dtype = None           # 数据类型：None=自动检测，Tesla T4/V100 用 float16，Ampere+ 用 bfloat16
load_in_4bit = True    # 是否使用 4bit 量化加载模型，可大幅节省显存（约 75%）

# ============ 1. 加载模型 ============
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/qwen2.5-7b-bnb-4bit",  # 预训练模型名称，支持 Qwen/Llama/Mistral 等
    max_seq_length=max_seq_length,             # 最大序列长度
    dtype=dtype,                               # 数据类型
    load_in_4bit=load_in_4bit,                 # 4bit 量化开关
    token=None,  # 明确设置 token 为 None
)

# ============ 2. 添加 LoRA 适配器 ============
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA 秩（rank），越大可学习参数越多，效果可能更好但显存占用更大，常用值：8/16/32/64
    target_modules=[         # 要应用 LoRA 的目标模块，这些是 Transformer 的核心层
        "q_proj",            # Query 投影层
        "k_proj",            # Key 投影层
        "v_proj",            # Value 投影层
        "o_proj",            # Output 投影层
        "gate_proj",         # FFN 门控层
        "up_proj",           # FFN 上投影层
        "down_proj"          # FFN 下投影层
    ],
    lora_alpha=32,           # LoRA 缩放因子，通常设为与 r 相同或 2 倍，影响学习率缩放
    lora_dropout=0,          # LoRA Dropout 率，Unsloth 优化后支持 0（无 dropout）
    bias="none",             # 是否训练偏置项："none"=不训练，"all"=全部训练，"lora_only"=仅 LoRA 层
    use_gradient_checkpointing="unsloth",  # 梯度检查点策略，"unsloth" 为优化版本，长序列必备，可节省 30% 显存
    random_state=3407,       # 随机种子，保证可复现性，3407 是一个常用的随机种子
)

# ============ 2.5 配置模版 ============

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5", # 或者 "chatml"，Qwen2.5 推荐使用这个
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}, # 对应你数据集中的字段
)

# ============ 3. 加载数据集 ============
# 使用 ChatML 格式的 JSONL 文件
dataset = load_dataset("json", data_files=SFT_PATH)

# ============ 4. 格式化数据 ============
def formatting_prompts_func(examples):
    """将数据转换为 Qwen 的 chat template 格式"""
    convos = examples["messages"]  # 获取对话消息列表
    texts = [
        tokenizer.apply_chat_template(
            convo,                       # 单条对话
            tokenize=False,              # 不进行分词，只返回格式化后的字符串
            add_generation_prompt=False  # 不添加生成提示（训练时不需要）
        ) 
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(
    formatting_prompts_func, 
    batched=True,  # 批量处理，提高效率
)

# ============ 5. 训练配置 ============
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,                          # 要训练的模型
    tokenizer=tokenizer,                  # 分词器
    train_dataset=dataset["train"],       # 训练数据集
    dataset_text_field="text",            # 数据集中文本字段的名称
    max_seq_length=max_seq_length,        # 最大序列长度
    dataset_num_proc=2,                   # 数据预处理的并行进程数
    packing=False,                        # 是否启用序列打包，多轮对话建议关闭以保持对话完整性
    
    args=TrainingArguments(
        # === 基础配置 ===
        output_dir="./outputs",           # 模型输出目录，保存 checkpoint 和日志
        per_device_train_batch_size=2,    # 每个 GPU 的批次大小，显存不足时减小
        gradient_accumulation_steps=4,    # 梯度累积步数，有效 batch_size = 2*4 = 8
        
        # === 训练轮次 ===
        num_train_epochs=3,               # 训练轮数，完整遍历数据集的次数
        max_steps=-1,                     # 最大训练步数，-1 表示使用 num_train_epochs
        
        # === 学习率配置 ===
        learning_rate=2e-4,               # 初始学习率，LoRA 微调常用 1e-4 ~ 5e-4
        warmup_steps=10,                  # 学习率预热步数，从 0 逐渐升到 learning_rate
        lr_scheduler_type="linear",       # 学习率调度器类型：linear/cosine/constant 等
        
        # === 优化器配置 ===
        optim="adamw_8bit",               # 优化器类型，adamw_8bit 是 Unsloth 优化的 8bit AdamW，节省显存
        weight_decay=0.01,                # 权重衰减（L2 正则化），防止过拟合
        
        # === 日志和保存 ===
        logging_steps=10,                 # 每隔多少步记录一次日志
        save_strategy="steps",            # 保存策略："steps"=按步数，"epoch"=按轮次
        save_steps=100,                   # 每隔多少步保存一次 checkpoint
        save_total_limit=3,               # 最多保留多少个 checkpoint，旧的会被删除
        
        # === 显存优化 ===
        fp16=not torch.cuda.is_bf16_supported(),  # 是否使用 FP16 混合精度（不支持 BF16 时启用）
        bf16=torch.cuda.is_bf16_supported(),      # 是否使用 BF16 混合精度（Ampere+ GPU 支持）
        gradient_checkpointing=True,              # 梯度检查点，用时间换显存
        
        # === 性能优化 ===
        dataloader_num_workers=2,         # 数据加载器的工作进程数
        group_by_length=True,             # 按序列长度分组，减少 padding，提高训练效率
        
        # === 其他 ===
        seed=3407,                        # 随机种子，保证可复现性
        report_to="tensorboard",          # 日志报告工具："tensorboard"/"wandb"/"none"
    ),
)

# ============ 6. 开始训练 ============
trainer_stats = trainer.train()

# ============ 7. 保存模型 ============
# 保存 LoRA 适配器（仅保存微调的参数，体积小）
model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")

# 保存合并后的完整模型（推理部署用）
model.save_pretrained_merged(
    "./merged_model",                     # 保存路径
    tokenizer,                            # 分词器
    save_method="merged_16bit",           # 保存方式："lora"=仅适配器，"merged_16bit"=合并后 FP16，"merged_4bit"=合并后 4bit
)

# ============ 8. 推理测试 ============
FastLanguageModel.for_inference(model)    # 切换到推理模式，速度提升约 2 倍

messages = [
    {"role": "user", "content": "你对创业怎么看？"}
]

inputs = tokenizer.apply_chat_template(
    messages,                             # 对话消息
    tokenize=True,                        # 进行分词
    add_generation_prompt=True,           # 添加生成提示（推理时需要）
    return_tensors="pt"                   # 返回 PyTorch 张量
).to("cuda")                              # 移动到 GPU

outputs = model.generate(
    input_ids=inputs,                     # 输入 token IDs
    max_new_tokens=512,                   # 最大生成 token 数
    temperature=0.7,                      # 温度参数，越高越随机，越低越确定（0.1~1.0）
    top_p=0.9,                            # 核采样参数，只从累积概率前 90% 的 token 中采样
    do_sample=True,                       # 是否启用采样，False 则使用贪婪解码
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # 解码并打印生成结果