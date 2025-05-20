import torch
from transformers import pipeline
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置随机种子以确保结果可重复
torch.manual_seed(123)

# 模型 ID
# model_id = "meta-llama/Llama-3.2-1B"
model_id = "meta-llama/Llama-3.1-8B-Instruct"  # 使用更大的模型进行测试

# 测试不同的线程数
thread_counts = list(range(3, 4))  # 从3到24
results = []

prompt = "The key to life is"

for num_threads in thread_counts:
    print(f"\n测试线程数: {num_threads}")
    # 设置线程数
    torch.set_num_threads(num_threads)
    print(torch.__config__.parallel_info())
    
    # 创建文本生成 pipeline
    start_load = time.time()
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省内存
        device_map="auto",          # 自动选择设备（CPU 或 GPU）
        max_new_tokens=256,         # 生成的最大 token 数
    )
    load_time = time.time() - start_load
    
    # 运行模型生成文本
    start_inference = time.time()
    res = pipe(prompt)
    inference_time = time.time() - start_inference
    total_time = load_time + inference_time
    
    # 保存结果
    results.append({
        "线程数": num_threads,
        "加载时间(秒)": load_time,
        "推理时间(秒)": inference_time,
        "总时间(秒)": total_time
    })
    
    # 输出结果和耗时
    print(f"线程数: {num_threads}, 加载时间: {load_time:.2f}秒, 推理时间: {inference_time:.2f}秒, 总时间: {total_time:.2f}秒")
    print(f"生成的文本: {res[0]['generated_text'][:100]}...")  # 只打印前100个字符

# # 保存结果到CSV文件
# results_df = pd.DataFrame(results)
# results_df.to_csv("llm_thread_benchmark.csv", index=False)
# print(f"\n结果已保存到 llm_thread_benchmark.csv")

# import pandas as pd
# import matplotlib.pyplot as plt

# # Read data from CSV file
# results_df = pd.read_csv("llm_thread_benchmark.csv")

# # Extract thread counts and time data
# thread_counts = results_df["线程数"].tolist()
# inference_times = results_df["推理时间(秒)"].tolist()
# total_times = results_df["总时间(秒)"].tolist()
# loading_times = results_df["加载时间(秒)"].tolist() if "加载时间(秒)" in results_df.columns else [t - i for t, i in zip(total_times, inference_times)]

# # Calculate speedup relative to 3 threads
# base_inference_time = inference_times[0]  # 3-thread inference time 
# base_total_time = total_times[0]          # 3-thread total time
# base_loading_time = loading_times[0]      # 3-thread loading time

# inference_speedups = [base_inference_time / time for time in inference_times]
# total_speedups = [base_total_time / time for time in total_times]
# loading_speedups = [base_loading_time / time for time in loading_times]

# # Increase font sizes for all text elements
# plt.rcParams.update({
#     'font.size': 14,
#     'axes.titlesize': 18,
#     'axes.labelsize': 16,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14,
# })

# # Create subplot layout
# plt.figure(figsize=(15, 10))

# # 1. Original time chart
# plt.subplot(2, 1, 1)
# plt.plot(thread_counts, inference_times, 'o-', linewidth=2, markersize=8, label='Inference Time')
# plt.plot(thread_counts, total_times, 'o-', linewidth=2, markersize=8, color='orange', label='Total Time')
# # plt.plot(thread_counts, loading_times, 'o-', linewidth=2, markersize=8, color='green', label='Loading Time')
# plt.title('Thread Count vs. LLM Performance', fontweight='bold')
# plt.xlabel('Thread Count', fontweight='bold')
# plt.ylabel('Time (seconds)', fontweight='bold')
# plt.grid(True)
# plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)

# # 2. Speedup chart
# plt.subplot(2, 1, 2)
# plt.plot(thread_counts, inference_speedups, 'o-', linewidth=2, markersize=8, label='Inference Time Speedup')
# plt.plot(thread_counts, total_speedups, 'o-', linewidth=2, markersize=8, color='orange', label='Total Time Speedup')
# # plt.plot(thread_counts, loading_speedups, 'o-', linewidth=2, markersize=8, color='green', label='Loading Time Speedup')
# plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (3 threads)')
# plt.title('Speedup Relative to 3 Threads', fontweight='bold')
# plt.xlabel('Thread Count', fontweight='bold')
# plt.ylabel('Speedup', fontweight='bold')
# plt.grid(True)
# plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)

# plt.tight_layout()
# plt.savefig('llm_thread_speedup_comparison.png', dpi=300)
# print(f"Speedup chart saved to llm_thread_speedup_comparison.png")

# # Add speedup data to the results DataFrame and save
# results_df['Inference Time Speedup'] = inference_speedups
# results_df['Total Time Speedup'] = total_speedups
# results_df['Loading Time Speedup'] = loading_speedups
# results_df.to_csv("llm_thread_benchmark_with_speedup.csv", index=False)
# print(f"Data with speedup metrics saved to llm_thread_benchmark_with_speedup.csv")