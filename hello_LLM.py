import torch
from transformers import pipeline
import time

# 设置随机种子以确保结果可重复
torch.manual_seed(123)

# 模型 ID
model_id = "meta-llama/Llama-3.2-1B"

# 创建文本生成 pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省内存
    device_map="auto",          # 自动选择设备（CPU 或 GPU）
    max_new_tokens=256,         # 生成的最大 token 数
)


# 运行模型生成文本
prompt = "The key to life is"
res = pipe(prompt)


# 输出结果和耗时
print("Generated text:", res[0]["generated_text"])
