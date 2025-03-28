import torch
from transformers import pipeline
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter  # 用于 TensorBoard 日志

# 设置随机种子以确保结果可重复
torch.manual_seed(123)

# 模型 ID
model_id = "meta-llama/Llama-3.2-1B"

# 创建文本生成 pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device="cpu",
    max_new_tokens=256,
)

# 输入提示
prompt = "The key to life is"

# 使用 PyTorch Profiler 分析
with profile(
    activities=[ProfilerActivity.CPU],
    # record_shapes=True,
    # profile_memory=True,
    # with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("log_dir")  # 直接写入 TensorBoard 日志
) as prof:
    with record_function("model_inference"):
        res = pipe(prompt)

# 输出生成结果
print("Generated text:", res[0]["generated_text"])

# 打印表格（可选）
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))