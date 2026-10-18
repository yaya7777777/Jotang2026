"""拓展：用 NumPy 手写的反向传播 + 最基础 SGD 训练完整 make_moons

结构: X -> Linear(2->16) -> ReLU -> Linear(16->1) -> Sigmoid -> BCE
绘制 loss / accuracy / 二维决策边界，并与 Task1 的 PyTorch 模型简单对比。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据
# ============================================================
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
X = X.astype(np.float64)
y = y.astype(np.float64)

# 划分 70/20/10
N = len(X)
idx = np.random.permutation(N)
n_tr, n_val = int(0.7 * N), int(0.2 * N)
tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]
Xtr, ytr = X[tr], y[tr]
Xva, yva = X[va], y[va]
Xte, yte = X[te], y[te]

# 标准化（只用训练集统计量）
mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
Xtr = (Xtr - mean) / std
Xva = (Xva - mean) / std
Xte = (Xte - mean) / std
Xs = (X - mean) / std   # 画图用

print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

# ============================================================
# 初始化参数
# ============================================================
D, H = 2, 16
W1 = np.random.randn(H, D) * 0.5
b1 = np.zeros(H)
W2 = np.random.randn(1, H) * 0.5
b2 = np.zeros(1)


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1.T + b1
    h = np.maximum(0, z1)
    z2 = h @ W2.T + b2
    p = 1 / (1 + np.exp(-z2))
    return z1, h, z2, p


def bce(p, y):
    y = y.reshape(-1, 1)
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def acc(X, y, W1, b1, W2, b2):
    _, _, _, p = forward(X, W1, b1, W2, b2)
    pred = (p.ravel() > 0.5).astype(float)
    return (pred == y).mean()


# ============================================================
# SGD 训练
# ============================================================
lr = 0.5
epochs = 200
hist_loss, hist_tr, hist_va = [], [], []

for ep in range(epochs):
    z1, h, z2, p = forward(Xtr, W1, b1, W2, b2)
    loss = bce(p, ytr)
    n = len(Xtr)

    # 反向
    dz2 = (p - ytr.reshape(-1, 1)) / n
    dW2 = dz2.T @ h
    db2 = dz2.sum(axis=0)
    dh = dz2 @ W2
    dz1 = dh * (z1 > 0)
    dW1 = dz1.T @ Xtr
    db1 = dz1.sum(axis=0)

    # SGD 更新
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    hist_loss.append(loss)
    hist_tr.append(acc(Xtr, ytr, W1, b1, W2, b2))
    hist_va.append(acc(Xva, yva, W1, b1, W2, b2))
    if (ep + 1) % 40 == 0:
        print(f"Epoch {ep+1:3d}/{epochs} | loss={loss:.4f} | "
              f"train={hist_tr[-1]:.4f} | val={hist_va[-1]:.4f}")

test_acc = acc(Xte, yte, W1, b1, W2, b2)
print(f"\nNumPy SGD Test Acc = {test_acc:.4f}")
print("(对比: Task1 PyTorch MLP Test Acc ≈ 0.98)")

# ============================================================
# 画图
# ============================================================
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(out, exist_ok=True)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(hist_loss, 'b-'); a1.set_title('NumPy SGD - Loss'); a1.set_xlabel('Epoch')
a2.plot(hist_tr, label='train'); a2.plot(hist_va, label='val')
a2.set_title('NumPy SGD - Accuracy'); a2.legend(); a2.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(os.path.join(out, 'numpy_sgd_curves.png'), dpi=150)
plt.close()

# 决策边界
xmin, xmax = Xs[:, 0].min() - .5, Xs[:, 0].max() + .5
ymin, ymax = Xs[:, 1].min() - .5, Xs[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(xmin, xmax, .02), np.arange(ymin, ymax, .02))
grid = np.c_[xx.ravel(), yy.ravel()]
_, _, _, pg = forward(grid, W1, b1, W2, b2)
Z = (pg > 0.5).astype(float).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=.3, cmap='coolwarm')
plt.scatter(Xs[y == 0, 0], Xs[y == 0, 1], c='tab:blue', edgecolor='k', s=30, label='Class 0')
plt.scatter(Xs[y == 1, 0], Xs[y == 1, 1], c='tab:orange', edgecolor='k', s=30, label='Class 1')
plt.title(f'NumPy SGD Decision Boundary (Test Acc={test_acc:.3f})')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(out, 'numpy_sgd_boundary.png'), dpi=150)
plt.close()
print("图已保存到 results/")
