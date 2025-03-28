import torch
from transformers import pipeline
import time
from torch.profiler import profile, record_function, ProfilerActivity

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

# 输入提示
prompt = "The key to life is"

# 使用 PyTorch Profiler 分析
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # 记录 CPU 和 GPU 活动
    record_shapes=True,  # 记录张量形状
    profile_memory=True,  # 记录内存使用情况
    with_stack=True      # 记录调用栈，便于调试
) as prof:
    with record_function("model_inference"):  # 标记推理部分
        res = pipe(prompt)

# 输出生成结果
print("Generated text:", res[0]["generated_text"])

# 打印 Profiler 结果
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))  # 按 CPU 时间排序
# 如果有 GPU，可以按 CUDA 时间排序
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 可选：导出为 Chrome Trace 文件，在浏览器中可视化
prof.export_chrome_trace("trace.json")