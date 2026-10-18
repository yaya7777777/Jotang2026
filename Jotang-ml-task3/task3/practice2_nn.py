"""实践 2：神经网络的梯度（两层二分类网络）

网络结构: X -> Linear(W1,b1) -> ReLU -> Linear(W2,b2) -> Sigmoid -> BCE Loss

NumPy 手写前向 + 反向，与 PyTorch autograd 逐项对照（形状、数值、最大绝对误差）。
先用单个样本验证，再换成一个小 batch 观察公式/形状变化。

形状标注（隐藏层 H=4，输入维 D=2）:
  单样本:
    X  (2,)        标签 y (1,)
    W1 (4,2)  b1 (4,)   z1=X@W1.T+b1 (4,)
    h  (4,)  = relu(z1)
    W2 (1,4)  b2 (1,)   z2=h@W2.T+b2 (1,)
    p  (1,)  = sigmoid(z2)
    loss (1,)  = BCE(p, y)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 数据：从 make_moons 取 1 个样本（后面再换 batch）
# ============================================================
X_all, y_all = make_moons(n_samples=1000, noise=0.15, random_state=42)


def run_case(X, y, tag):
    """对给定输入跑 NumPy 前向+反向 + PyTorch 对照"""
    print(f"\n{'='*60}\n{tag}\n{'='*60}")
    X = np.atleast_2d(X).astype(np.float64)   # (N, 2)
    y = np.atleast_1d(y).astype(np.float64)   # (N,)
    N, D = X.shape
    H = 4                                     # 隐藏层神经元数

    # ---------- 初始化参数（与 PyTorch 用同一份数值）----------
    W1 = np.random.randn(H, D) * 0.5          # (H, D)
    b1 = np.zeros(H)                          # (H,)
    W2 = np.random.randn(1, H) * 0.5          # (1, H)
    b2 = np.zeros(1)                          # (1,)

    # ================= NumPy 前向 =================
    z1 = X @ W1.T + b1                        # (N,H)
    h = np.maximum(0, z1)                     # ReLU (N,H)
    z2 = h @ W2.T + b2                        # (N,1)
    p = 1 / (1 + np.exp(-z2))                 # Sigmoid (N,1)
    y_col = y.reshape(-1, 1)                  # (N,1)
    eps = 1e-12
    # 二元交叉熵（对每个样本求 mean）
    loss = -np.mean(y_col * np.log(p + eps) + (1 - y_col) * np.log(1 - p + eps))

    print(f"前向: z1{z1.shape} h{h.shape} z2{z2.shape} p{p.shape} loss={loss:.6f}")

    # ================= NumPy 反向（链式法则） =================
    # 关键简化: BCE + Sigmoid 组合 -> dloss/dz2 = (p - y)/N
    dz2 = (p - y_col) / N                     # (N,1)
    dW2 = dz2.T @ h                           # (1,N)@(N,H) = (1,H)
    db2 = dz2.sum(axis=0)                     # (1,)
    dh = dz2 @ W2                             # (N,1)@(1,H) = (N,H)
    dz1 = dh * (z1 > 0)                       # ReLU 掩码 (N,H)
    dW1 = dz1.T @ X                           # (H,N)@(N,D) = (H,D)
    db1 = dz1.sum(axis=0)                     # (H,)

    # ================= PyTorch 前向 + autograd =================
    tX = torch.tensor(X)
    tW1 = torch.tensor(W1, requires_grad=True)
    tb1 = torch.tensor(b1, requires_grad=True)
    tW2 = torch.tensor(W2, requires_grad=True)
    tb2 = torch.tensor(b2, requires_grad=True)

    tz1 = tX @ tW1.T + tb1
    th = torch.relu(tz1)
    tz2 = th @ tW2.T + tb2
    tp = torch.sigmoid(tz2)
    tloss = nn.functional.binary_cross_entropy(tp, torch.tensor(y_col))
    tloss.backward()

    # ================= 逐项对照 =================
    pairs = [
        ("W1", dW1, tW1.grad.numpy()),
        ("b1", db1, tb1.grad.numpy()),
        ("W2", dW2, tW2.grad.numpy()),
        ("b2", db2, tb2.grad.numpy()),
    ]
    print(f"\n{'参数':<6}{'形状':<12}{'最大绝对误差':<14}一致?")
    max_err_all = 0.0
    for name, g_np, g_pt in pairs:
        err = np.max(np.abs(g_np - g_pt))
        max_err_all = max(max_err_all, err)
        ok = "✓" if err < 1e-10 else "✗"
        print(f"{name:<6}{str(g_np.shape):<12}{err:<14.2e}{ok}")
    print(f"\nloss: NumPy={loss:.8f}  PyTorch={tloss.item():.8f}")
    print(f"全部参数最大绝对误差 = {max_err_all:.2e}  ->",
          "梯度一致!" if max_err_all < 1e-10 else "梯度不一致!")
    return W1, b1, W2, b2


# ---------- 单样本 ----------
run_case(X_all[0], y_all[0], "情况 1：单个样本 (N=1)")

# ---------- 小 batch ----------
run_case(X_all[:8], y_all[:8], "情况 2：小 batch (N=8)")
