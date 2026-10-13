
import os
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from moons_loader import MoonsDataset

# 固定随机种子，保证可复现
SEED = 42

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 可修改参数 MLP
ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
}

def build_mlp(in_size=2, hidden_sizes=(16, 8), num_classes=2,
              activation='relu'):

    act_fn = ACTIVATIONS[activation]
    layers = []
    prev = in_size
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn())
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)



 # 数据加载（带noise）

def load_data(n_samples=1000, noise=0.15, batch_size=14, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = MoonsDataset(n_samples=n_samples, noise=noise, random_state=seed)

    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val

    g = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=g)

    # 标准化只用训练集统计量
    train_data = dataset.data[train_set.indices]
    mean = train_data.mean(dim=0)
    std = train_data.std(dim=0)
    dataset.data = (dataset.data - mean) / std

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    return dataset, train_loader, val_loader, test_loader


# 训练函数（记录时间 + 显存）
def train_model(model, train_loader, val_loader, device,
                lr=0.005, epochs=100, optimizer_name='adam',
                save_path=None, verbose=True):

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    opt_class = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW,
    }[optimizer_name]
    optimizer = opt_class(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()# 交叉熵损失函数（多分类）
    
    # 重置显存计数器
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    start = time.time()
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0.0

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
        val_acc = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / total

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Loss {avg_loss:.4f} | "
                  f"Train {train_acc:.4f} | Val {val_acc:.4f}")

    elapsed = time.time() - start

    # 峰值显存（MB）
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        peak_mem = 0.0

    # 保存权重
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

    if verbose:
        print(f"    Time: {elapsed:.2f}s | Peak GPU Mem: {peak_mem:.1f}MB")

    return model, history, elapsed, peak_mem


def evaluate(model, data_loader, device):
    # 计算准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total




# 对比实验运行
def run_experiment(exp_name, configs, base_config,
                   train_loader, val_loader, test_loader, device, save_dir):
    print(f"\n--- {exp_name} ---")
    results = []

    for label, overrides in configs:
        cfg = {**base_config, **overrides}
        print(f"  [{label}]")

        # 如果改了batch_size或noise，重新加载数据
        cur_train, cur_val, cur_test = train_loader, val_loader, test_loader
        if 'batch_size' in overrides or 'noise' in overrides:
            bs = cfg.get('batch_size', 14)
            noise = cfg.get('noise', 0.15)
            _, cur_train, cur_val, cur_test = load_data(
                n_samples=1000, noise=noise, batch_size=bs, seed=SEED)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(SEED)

        model = build_mlp(
            in_size=2,
            hidden_sizes=cfg['hidden_sizes'],
            activation=cfg['activation'],
        ).to(device)

        model, hist, elapsed, peak_mem = train_model(
            model, cur_train, cur_val, device,
            lr=cfg['lr'], epochs=cfg['epochs'],
            optimizer_name=cfg['optimizer'],
            save_path=None, verbose=True,
        )
        test_acc = evaluate(model, cur_test, device)
        n_params = sum(p.numel() for p in model.parameters())

        results.append({
            'label': label,
            'history': hist,
            'test_acc': test_acc,
            'time': elapsed,
            'mem': peak_mem,
            'n_params': n_params,
        })
        print(f"    -> Test Acc {test_acc:.4f} | Params {n_params} | "
              f"Time {elapsed:.2f}s | Mem {peak_mem:.1f}MB")

    plot_experiment(exp_name, results, save_dir)
    return results


def plot_experiment(exp_name, results, save_dir):
    # 为一个实验画对比图
    n = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 左：loss 曲线
    ax = axes[0]
    for r in results:
        ax.plot(r['history']['train_loss'], label=r['label'], alpha=0.8)
    ax.set_title(f'{exp_name} - Training Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()

    # 中：val acc 曲线
    ax = axes[1]
    for r in results:
        ax.plot(r['history']['val_acc'], label=r['label'], alpha=0.8)
    ax.set_title(f'{exp_name} - Validation Accuracy')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.legend()

    # 右：Test Acc / Time / Mem 柱状图
    ax = axes[2]
    labels = [r['label'] for r in results]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, [r['test_acc'] for r in results], width, label='Test Acc', color='tab:blue')
    ax.bar(x, [r['time'] for r in results], width, label='Time (s)', color='tab:orange')
    ax2 = ax.twinx()
    ax2.bar(x + width, [r['mem'] for r in results], width, label='Mem (MB)', color='tab:green')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_title(f'{exp_name} - Metrics')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax.set_ylabel('Acc / Time'); ax2.set_ylabel('Mem (MB)')

    plt.tight_layout()
    fname = exp_name.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(save_dir, f'{fname}.png'), dpi=150)
    plt.close()


# 可视化函数

def visualize_data(data, labels, save_dir, title="make_moons Dataset"):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[labels == 0, 0], data[labels == 0, 1],
                c='tab:blue', edgecolor='k', s=40, label='Class 0')
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1],
                c='tab:orange', edgecolor='k', s=40, label='Class 1')
    plt.title(title); plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'raw_data.png'), dpi=150); plt.close()

def visualize_training(history, losses, save_dir):
    epochs = len(history['train_acc'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(range(1, epochs+1), losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
    ax1.set_title('Training Loss')
    ax2.plot(range(1, epochs+1), history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(range(1, epochs+1), history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
    ax2.set_title('Train & Val Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_acc_curves.png'), dpi=150); plt.close()

def visualize_decision_boundary(model, data, labels, device, save_dir):
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        Z = model(grid).argmax(dim=1).cpu().numpy().reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(data[labels == 0, 0], data[labels == 0, 1],
                c='tab:blue', edgecolor='k', s=30, label='Class 0')
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1],
                c='tab:orange', edgecolor='k', s=30, label='Class 1')
    plt.title('Decision Boundary'); plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'decision_boundary.png'), dpi=150); plt.close()

def visualize_confusion_matrix(model, test_loader, device, save_dir):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            preds = model(data).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1']); ax.set_yticklabels(['True 0', 'True 1'])
    ax.set_title('Confusion Matrix'); fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150); plt.close()

    wrong = np.array(all_preds) != np.array(all_labels)
    print(f"  Misclassified: {wrong.sum()} / {len(all_labels)}")
    for idx in np.where(wrong)[0][:5]:
        print(f"    Sample {idx}: true={all_labels[idx]}, pred={all_preds[idx]}")
    return cm

def visualize_misclassified(model, test_loader, device, save_dir):
    model.eval()
    all_data, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            preds = model(data).argmax(dim=1).cpu().numpy()
            all_data.append(data.cpu().numpy()[0])
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_data = np.array(all_data); all_preds = np.array(all_preds); all_labels = np.array(all_labels)
    wrong = all_preds != all_labels
    plt.figure(figsize=(8, 6))
    plt.scatter(all_data[~wrong, 0], all_data[~wrong, 1],
                c=all_labels[~wrong], cmap='viridis', edgecolor='k', s=30, label='Correct')
    plt.scatter(all_data[wrong, 0], all_data[wrong, 1],
                c='red', marker='X', s=120, label='Misclassified', zorder=5)
    plt.title(f'Misclassified Samples ({wrong.sum()} errors)')
    plt.xlabel('Feature 1'); plt.ylabel('Feature 2'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'misclassified.png'), dpi=150); plt.close()




# 主函数
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 基础配置（对照组）
    BASE = {
        'hidden_sizes': (16, 8),
        'activation': 'relu',
        'lr': 0.005,
        'epochs': 100,
        'optimizer': 'adam',
    }
    EPOCHS_EXP = 50  # 实验用更少 epoch 加速


    dataset, train_loader, val_loader, test_loader = load_data(
        n_samples=1000, noise=0.15, batch_size=14, seed=SEED)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # 数据可视化
    all_data = dataset.data.numpy()
    all_labels = dataset.labels.numpy()
    visualize_data(all_data, all_labels, results_dir)

    # 训练
    torch.manual_seed(SEED)
    model = build_mlp(in_size=2, hidden_sizes=BASE['hidden_sizes'],
                      activation=BASE['activation']).to(device)
    model, history, elapsed, peak_mem = train_model(
        model, train_loader, val_loader, device,
        lr=BASE['lr'], epochs=BASE['epochs'],
        optimizer_name=BASE['optimizer'],
        save_path=os.path.join(base_dir, 'weights', 'nn.pth'))
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    visualize_training(history, history['train_loss'], results_dir)
    visualize_decision_boundary(model, all_data, all_labels, device, results_dir)

    cm = visualize_confusion_matrix(model, test_loader, device, results_dir)
    print(f"  {cm}")
    visualize_misclassified(model, test_loader, device, results_dir)

    # 模型加载
    m2 = build_mlp(hidden_sizes=BASE['hidden_sizes'], activation=BASE['activation']).to(device)
    m2.load_state_dict(torch.load(os.path.join(base_dir, 'weights', 'nn.pth'),
                                  map_location=device, weights_only=True))
    print(f"  Loaded Test Acc: {evaluate(m2, test_loader, device):.4f}")


    # 对照实验
    print("\n=== Controlled Experiments (7 groups) ===")
    base_exp = {**BASE, 'epochs': EPOCHS_EXP}

    all_results = {}

    # 实验 1：隐藏层宽度
    all_results['width'] = run_experiment(
        'Hidden Width', [
            ('width=4', {'hidden_sizes': (4,)}),
            ('width=8', {'hidden_sizes': (8,)}),
            ('width=16', {'hidden_sizes': (16,)}),
            ('width=32', {'hidden_sizes': (32,)}),
            ('width=64', {'hidden_sizes': (64,)}),
        ], base_exp, train_loader, val_loader, test_loader, device, results_dir)

    # 实验 2：隐藏层数（深度）
    all_results['depth'] = run_experiment(
        'Depth', [
            ('1 layer', {'hidden_sizes': (16,)}),
            ('2 layers', {'hidden_sizes': (16, 8)}),
            ('3 layers', {'hidden_sizes': (16, 16, 8)}),
            ('4 layers', {'hidden_sizes': (32, 16, 16, 8)}),
        ], base_exp, train_loader, val_loader, test_loader, device, results_dir)

    # 实验 3：激活函数
    all_results['activation'] = run_experiment(
        'Activation', [
            ('ReLU', {'activation': 'relu'}),
            ('Tanh', {'activation': 'tanh'}),
            ('Sigmoid', {'activation': 'sigmoid'}),
        ], base_exp, train_loader, val_loader, test_loader, device, results_dir)

    # 实验 4：学习率
    all_results['lr'] = run_experiment(
        'Learning Rate', [
            ('lr=0.001', {'lr': 0.001}),
            ('lr=0.005', {'lr': 0.005}),
            ('lr=0.01', {'lr': 0.01}),
            ('lr=0.05', {'lr': 0.05}),
            ('lr=0.1', {'lr': 0.1}),
        ], base_exp, train_loader, val_loader, test_loader, device, results_dir)

    # 实验 5：优化器
    all_results['optimizer'] = run_experiment(
        'Optimizer', [
            ('SGD', {'optimizer': 'sgd'}),
            ('Adam', {'optimizer': 'adam'}),
            ('AdamW', {'optimizer': 'adamw'}),
        ], base_exp, train_loader, val_loader, test_loader, device, results_dir)

    # 实验 6：batch size
    all_results['batch'] = run_experiment(
        'Batch Size', [
            ('bs=8', {'batch_size': 8}),
            ('bs=16', {'batch_size': 16}),
            ('bs=32', {'batch_size': 32}),
            ('bs=64', {'batch_size': 64}),
            ('bs=128', {'batch_size': 128}),
        ], base_exp, train_loader, val_loader, test_loader, device, results_dir)

    # 实验 7：数据噪声
    all_results['noise'] = run_experiment(
        'Data Noise', [
            ('noise=0.0', {'noise': 0.0}),
            ('noise=0.1', {'noise': 0.1}),
            ('noise=0.2', {'noise': 0.2}),
            ('noise=0.3', {'noise': 0.3}),
        ], base_exp, train_loader, val_loader, test_loader, device, results_dir)

   
    # 汇总实验结果
    print("\n=== Experiment Summary ===")
    print(f"{'Experiment':<20} {'Config':<15} {'Test Acc':<10} {'Time(s)':<10} {'Mem(MB)':<10} {'Params':<10}")
    print("-" * 75)
    for exp_name, results in all_results.items():
        for r in results:
            print(f"{exp_name:<20} {r['label']:<15} {r['test_acc']:<10.4f} "
                  f"{r['time']:<10.2f} {r['mem']:<10.1f} {r['n_params']:<10}")


if __name__ == '__main__':
    main()
