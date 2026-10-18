"""拓展 1：NumPy 手写最小单头 Scaled Dot-Product Attention
随机输入，输出 Attention 分数矩阵、权重矩阵并画成热力图
"""
import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 随机输入：序列长度 6，模型维度 16
N, D = 6, 16
X = np.random.randn(N, D)                     # (N, D)

# 三个独立的线性变换（论文中的 W_Q / W_K / W_V）
W_Q = np.random.randn(D, D) * 0.3
W_K = np.random.randn(D, D) * 0.3
W_V = np.random.randn(D, D) * 0.3
Q, K, V = X @ W_Q, X @ W_K, X @ W_V           # (N, D)

# Scaled Dot-Product Attention 核心三步
raw = Q @ K.T                                 # 未缩放点积 (N, N)
scores = raw / np.sqrt(D)                     # 缩放 (N, N)
weights = np.exp(scores - scores.max(axis=1, keepdims=True))
weights /= weights.sum(axis=1, keepdims=True) # 逐行 softmax (N, N)
output = weights @ V                          # 加权求和 (N, D)

# 验证"为什么要除 sqrt(d_k)"：未缩放分数的标准差应约为 sqrt(d_k)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"未缩放点积分数 std = {raw.std():.3f}  ≈ sqrt(d_k) = {np.sqrt(D):.3f}")
print(f"缩放后分数 std = {scores.std():.3f}（回到 ~1 量级，softmax 不会饱和）\n")

print("Attention 分数矩阵 scores = QK^T / sqrt(d_k):")
print(scores)
print("\nAttention 权重矩阵 weights = softmax(scores)，每行是一个 query 的分布:")
print(weights)
print("\n每行权重之和 =", weights.sum(axis=1).round(6))
print("输出 output = weights @ V，形状:", output.shape)

# 热力图：左分数、右权重
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(out, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, mat, title, cmap, fmt in [
        (axes[0], scores, 'Scores = QK$^T$/√d_k', 'coolwarm', '{:+.1f}'),
        (axes[1], weights, 'Weights = softmax(Scores)', 'viridis', '{:.2f}')]:
    im = ax.imshow(mat, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_title(title); ax.set_xlabel('Key（被看的位置）'); ax.set_ylabel('Query（当前词）')
    for i in range(N):
        for j in range(N):
            ax.text(j, i, fmt.format(mat[i, j]), ha='center', va='center',
                    fontsize=7, color='w' if cmap == 'viridis' else 'k')
plt.tight_layout()
plt.savefig(os.path.join(out, 'single_head_attention.png'), dpi=150)
plt.close()

