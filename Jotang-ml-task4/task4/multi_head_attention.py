"""拓展 2：手写多头注意力（在拓展 1 基础上拆成 H 个头）
每头有独立的 W_Q/W_K/W_V，在低维子空间各自做注意力
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

N, D, H = 6, 16, 4                            # 序列长度 / 模型维度 / 头数
d_k = D // H                                  # 每头维度 4

X = np.random.randn(N, D)                     # (N, D)

def softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

# 每头一套独立的投影矩阵；给不同头不同的缩放系数，
# 让各头 Q/K 的"尖锐程度"不同，方便观察多头模式差异
head_outputs, head_weights = [], []
for i in range(H):
    s = 0.15 + 0.15 * i                       # 头 0 平缓 -> 头 3 尖锐
    Wq = np.random.randn(D, d_k) * s
    Wk = np.random.randn(D, d_k) * s
    Wv = np.random.randn(D, d_k) * 0.3
    Q, K, V = X @ Wq, X @ Wk, X @ Wv          # (N, d_k)
    w = softmax(Q @ K.T / np.sqrt(d_k))       # (N, N) 每头自己的注意力
    head_outputs.append(w @ V)                # (N, d_k)
    head_weights.append(w)

concat = np.concatenate(head_outputs, axis=1) # (N, D) 拼回 16 维
W_O = np.random.randn(D, D) * 0.3             # 输出融合投影
output = concat @ W_O                         # (N, D)

np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"输入 X: {X.shape}  ->  {H} 个头各输出 (N, d_k)={N, d_k}")
print("拼接后 concat 形状:", concat.shape, "-> 过 W_O 后输出形状:", output.shape)
print("\n各头在 query=0 这一行的权重分布（各头关注的位置明显不同）:")
for i, w in enumerate(head_weights):
    print(f"  头{i}: {w[0]}   (max={w[0].max():.2f})")
print("\n各头权重分布的尖锐程度（熵，越小越集中）:")
for i, w in enumerate(head_weights):
    ent = -(w * np.log(w + 1e-12)).sum(axis=1).mean()
    print(f"  头{i}: 平均熵 = {ent:.3f}  (均匀分布应为 {np.log(N):.3f})")

# 每头热力图
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(out, exist_ok=True)

fig, axes = plt.subplots(1, H, figsize=(4.2 * H, 4))
for i, (ax, w) in enumerate(zip(axes, head_weights)):
    im = ax.imshow(w, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'Head {i} Weights')
    ax.set_xlabel('Key'); ax.set_ylabel('Query')
    for r in range(N):
        for c in range(N):
            ax.text(c, r, f'{w[r, c]:.2f}', ha='center', va='center',
                    fontsize=6, color='w')
plt.tight_layout()
plt.savefig(os.path.join(out, 'multi_head_attention.png'), dpi=150)
plt.close()

