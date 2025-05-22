import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")  # 忽略无关警告

# 模型和分词器路径
model_id = "meta-llama/Meta-Llama-3.1-8B"  # 或 Meta-Llama-3-8B
# hf_token = "your_hf_token_here"  # 替换为你的 HF 令牌

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # token=hf_token,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 遍历模型层，提取权重矩阵维度
print("\n权重矩阵维度:")
for name, param in model.named_parameters():
    if param.requires_grad:  # 仅打印可训练参数
        print(f"层: {name}, 维度: {param.shape}")

# 打印模型配置
config = model.config
print("模型配置:")
print(f"词汇表大小 (vocab_size): {config.vocab_size}")
print(f"隐藏维度 (hidden_size): {config.hidden_size}")
print(f"中间维度 (intermediate_size): {config.intermediate_size}")
print(f"层数 (num_hidden_layers): {config.num_hidden_layers}")
print(f"注意力头数 (num_attention_heads): {config.num_attention_heads}")
print(f"键/值头数 (num_key_value_heads): {config.num_key_value_heads}")
print(f"最大序列长度 (max_position_embeddings): {config.max_position_embeddings}")



# 计算典型 GEMM 规模
batch_size = 32
seq_length = 512
hidden_size = config.hidden_size
intermediate_size = config.intermediate_size
vocab_size = config.vocab_size
head_dim = hidden_size // config.num_attention_heads

print("\n典型 GEMM 规模 (batch_size=32, seq_length=512):")
# 嵌入层
print(f"输入嵌入: [{batch_size * seq_length}, {vocab_size}, {hidden_size}]")
print(f"输出嵌入: [{batch_size * seq_length}, {hidden_size}, {vocab_size}]")
# 自注意力
print(f"Q·K^T (单头): [{batch_size * seq_length}, {head_dim}, {batch_size * seq_length}]")
print(f"Attention·V: [{batch_size * seq_length}, {batch_size * seq_length}, {hidden_size}]")
print(f"自注意力输出线性层: [{batch_size * seq_length}, {hidden_size}, {hidden_size}]")
# FFN
print(f"FFN 第一个线性层: [{batch_size * seq_length}, {hidden_size}, {intermediate_size}]")
print(f"FFN 第二个线性层: [{batch_size * seq_length}, {intermediate_size}, {hidden_size}]")

# 示例推理，验证矩阵使用
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model(**inputs)
print("\n推理输出 logits 形状:", outputs.logits.shape)  # [batch_size, seq_length, vocab_size]