import torch
from transformers import pipeline
from torch.profiler import profile, record_function, ProfilerActivity

# 设置随机种子以确保结果可重复
torch.manual_seed(123)

# 模型 ID
model_id = "meta-llama/Llama-3.2-1B"

# 创建文本生成 pipeline（明确指定设备为 CPU）
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省内存
    device="cpu",                # 强制使用 CPU
    max_new_tokens=256,          # 生成的最大 token 数
)

# 输入提示
prompt = "The key to life is"

# 使用 PyTorch Profiler 分析 CPU 性能
with profile(
    activities=[ProfilerActivity.CPU],  # 只记录 CPU 活动
    record_shapes=True,                 # 记录张量形状
    profile_memory=True,                # 记录内存使用情况
    with_stack=True                     # 记录调用栈
) as prof:
    with record_function("model_inference"):  # 标记推理部分
        res = pipe(prompt)

# 输出生成结果
print("Generated text:", res[0]["generated_text"])

# 打印 CPU 性能分析结果
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# 可选：按内存使用排序
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# 导出为 Chrome Trace 文件以可视化
prof.export_chrome_trace("cpu_trace.json")