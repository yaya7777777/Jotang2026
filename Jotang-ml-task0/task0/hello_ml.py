
import sys
import torch

# 自动选择设备：优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current computing device: {device}")
if device.type == 'cuda':
    print(f"GPU model: {torch.cuda.get_device_name(0)}")

# 输出 Python 与 PyTorch 版本
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")

# 张量运算演示：模拟全连接层 z = xW^T + b
x = torch.randn(2, 3)   # 输入矩阵：batch=2, 输入特征=3
W = torch.randn(4, 3)   # 权重矩阵：输出神经元=4, 输入特征=3
b = torch.randn(4)      # 偏置向量：自动广播到每个样本
z = x @ W.T + b

print(f"Input x shape: {tuple(x.shape)}")
print(f"Weight W shape: {tuple(W.shape)}")

print(f"FC layer output z = xW^T + b, shape: {tuple(z.shape)}")
print(f"z =\n{z}")
