"""额外实验：类别不均衡 + 故意过拟合"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from moons_loader import MoonsDataset

SEED = 42
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_mlp(in_size=2, hidden_sizes=(16, 8), num_classes=2, activation='relu'):
    act_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}[activation]
    layers = []
    prev = in_size
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn())
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0; total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)

def train_model(model, train_loader, val_loader, device, lr=0.005, epochs=100, verbose=True):
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    for epoch in range(epochs):
        model.train()
        correct = 0; total = 0; epoch_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.squeeze(-1)
            total += labels.size(0)
            optimizer.zero_grad()
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * labels.size(0)
            loss.backward()
            optimizer.step()
        train_acc = correct / total
        val_acc, _, _ = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / total
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Loss {avg_loss:.4f} | "
                  f"Train {train_acc:.4f} | Val {val_acc:.4f}")
    elapsed = time.time() - start
    if verbose:
        print(f"    Time: {elapsed:.2f}s")
    return model, history, elapsed

import time

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# ============================================================
# 实验 6：类别不均衡
# ============================================================
print("=== 类别不均衡实验 ===")

torch.manual_seed(SEED); np.random.seed(SEED)

# 生成不均衡数据：class 0 占 90%，class 1 占 10%
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.15, random_state=SEED)
idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]
# 保留全部 class 0，但只保留 1/9 的 class 1
np.random.seed(SEED)
keep_1 = np.random.choice(idx_1, size=10, replace=False)
X_imb = np.vstack([X[idx_0], X[keep_1]])
y_imb = np.concatenate([y[idx_0], y[keep_1]])

print(f"不均衡数据: class 0 = {(y_imb==0).sum()}, class 1 = {(y_imb==1).sum()}")
print(f"比例: 0:{(y_imb==0).sum()/(y_imb==1).sum():.1f}:1")

# 划分
n = len(X_imb)
n_train = int(0.7 * n); n_val = int(0.2 * n); n_test = n - n_train - n_val
g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(n, generator=g).numpy()
X_train, y_train = X_imb[perm[:n_train]], y_imb[perm[:n_train]]
X_val, y_val = X_imb[perm[n_train:n_train+n_val]], y_imb[perm[n_train:n_train+n_val]]
X_test, y_test = X_imb[perm[n_train+n_val:]], y_imb[perm[n_train+n_val:]]

# 标准化
mean = X_train.mean(axis=0); std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# DataLoader
train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long))
val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                        torch.tensor(y_val, dtype=torch.long))
test_ds = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                         torch.tensor(y_test, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

# 训练
model_imb = build_mlp(in_size=2, hidden_sizes=(16, 8)).to(device)
model_imb, hist_imb, _ = train_model(model_imb, train_loader, val_loader, device,
                                      lr=0.005, epochs=100, verbose=True)
test_acc, preds, labels = evaluate(model_imb, test_loader, device)
print(f"Imbalanced Test Acc: {test_acc:.4f}")

# 混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, preds)
print(f"Confusion Matrix:\n{cm}")

# 计算各指标
tn, fp, fn, tp = cm.ravel()
print(f"Precision (class 1): {tp/(tp+fp):.4f}")
print(f"Recall (class 1): {tp/(tp+fn):.4f}")
print(f"F1 Score (class 1): {2*tp/(2*tp+fp+fn):.4f}")

# 画混淆矩阵图
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred 0', 'Pred 1']); ax.set_yticklabels(['True 0', 'True 1'])
ax.set_title(f'Imbalanced Data Confusion Matrix\n(90% vs 10%)')
fig.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'imbalanced_cm.png'), dpi=150); plt.close()

# 画训练曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(hist_imb['train_loss'], 'b-', label='Train Loss')
ax1.set_title('Imbalanced Training Loss'); ax1.legend()
ax2.plot(hist_imb['train_acc'], 'b-', label='Train Acc')
ax2.plot(hist_imb['val_acc'], 'r-', label='Val Acc')
ax2.set_title('Imbalanced Accuracy'); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'imbalanced_curves.png'), dpi=150); plt.close()

print("Imbalanced experiment done.\n")

# ============================================================
# 实验 7：故意过拟合
# ============================================================
print("=== 故意过拟合实验 ===")

torch.manual_seed(SEED); np.random.seed(SEED)

# 用原始 make_moons 数据，但只用很小的训练集（100 个样本），模型极大
dataset = MoonsDataset(n_samples=1000, noise=0.15, random_state=SEED)
n = len(dataset)
n_train = 100  # 故意用很少的训练样本
n_val = 200
n_test = n - n_train - n_val

g = torch.Generator().manual_seed(SEED)
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [n_train, n_val, n_test], generator=g)

train_data = dataset.data[train_set.indices]
mean = train_data.mean(dim=0); std = train_data.std(dim=0)
dataset.data = (dataset.data - mean) / std

train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

# 极大模型：5 层 256 宽度
model_over = build_mlp(in_size=2, hidden_sizes=(256, 256, 128, 128, 64)).to(device)
n_params = sum(p.numel() for p in model_over.parameters())
print(f"Overfit model params: {n_params} (vs normal 202)")

model_over, hist_over, _ = train_model(model_over, train_loader, val_loader, device,
                                        lr=0.001, epochs=500, verbose=True)
train_acc, _, _ = evaluate(model_over, train_loader, device)
val_acc, _, _ = evaluate(model_over, val_loader, device)
test_acc, _, _ = evaluate(model_over, test_loader, device)
print(f"Overfit Train Acc: {train_acc:.4f}")
print(f"Overfit Val Acc:   {val_acc:.4f}")
print(f"Overfit Test Acc:  {test_acc:.4f}")

# 画图：训练 loss 持续下降，但 val loss 先降后升（U 型拐点）
epochs = len(hist_over['train_loss'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, epochs+1), hist_over['train_loss'], 'b-', label='Train Loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
ax1.set_title('Overfit - Train Loss (持续下降)')

ax2.plot(range(1, epochs+1), hist_over['train_acc'], 'b-', label='Train Acc')
ax2.plot(range(1, epochs+1), hist_over['val_acc'], 'r-', label='Val Acc')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
ax2.set_title('Overfit - Train vs Val Accuracy\n(训练100%, 验证低)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'overfit_curves.png'), dpi=150); plt.close()

# 决策边界（应该很扭曲）
all_data = dataset.data.numpy()
all_labels = dataset.labels.numpy()
# 直接画决策边界
x_min, x_max = all_data[:, 0].min() - 0.5, all_data[:, 0].max() + 0.5
y_min, y_max = all_data[:, 1].min() - 0.5, all_data[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
model_over.eval()
with torch.no_grad():
    Z = model_over(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(all_data[all_labels == 0, 0], all_data[all_labels == 0, 1],
            c='tab:blue', edgecolor='k', s=30, label='Class 0')
plt.scatter(all_data[all_labels == 1, 0], all_data[all_labels == 1, 1],
            c='tab:orange', edgecolor='k', s=30, label='Class 1')
plt.title(f'Overfit Decision Boundary\n(5-layer x 256 width, 100 train samples)')
plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'overfit_boundary.png'), dpi=150); plt.close()

print("Overfit experiment done.")
