"""实验2：猫狗分类 CNN（Kaggle Dogs vs Cats 真实数据）

数据布局：
  ../train/cat.0.jpg ... cat.12499.jpg, dog.0.jpg ... dog.12499.jpg
  ../test1/1.jpg ... 12500.jpg   (Kaggle 官方测试集，无标签，仅用于推理演示)
从 train 中按 8:1:1 划分训练/验证/测试。
"""
import os, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# 数据读取与预处理（划分/增强/DataLoader/防乱码
from loader_data import SEED, BASE, train_tf, eval_tf, build_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'weights'), exist_ok=True)


# CNN
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 卷积层
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)               # 池化层（最大池化）
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # 64->32
        x = self.pool(nn.functional.relu(self.conv2(x)))  # 32->16
        x = self.pool(nn.functional.relu(self.conv3(x)))  # 16->8
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_loader, val_loader, test_loader, train_set, val_set, test_set = build_loaders()
    print(f"全部数据: cat=12500, dog=12500")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    model = CatDogCNN().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # 一张图经过各层后的形状
    x0 = torch.randn(1, 3, 64, 64).to(device)
    for name, layer in [('conv1', model.conv1), ('pool1', model.pool),
                        ('conv2', model.conv2), ('pool2', model.pool),
                        ('conv3', model.conv3), ('pool3', model.pool)]:
        x0 = layer(x0)
        print(f"{name}: {tuple(x0.shape)}")

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 10
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    start = time.time()
    for epoch in range(EPOCHS):
        model.train(); correct = total = 0; epoch_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item(); total += labels.size(0)
        train_acc = correct / total

        model.eval(); correct = total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                correct += (model(data).argmax(1) == labels).sum().item(); total += labels.size(0)
        val_acc = correct / total
        history['train_loss'].append(epoch_loss / len(train_set))
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(BASE, 'weights', 'best.pth'))
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss {epoch_loss/len(train_set):.4f} | "
              f"Train {train_acc:.4f} | Val {val_acc:.4f}")
    print(f"Time: {time.time()-start:.1f}s | Best Val: {best_acc:.4f}")
    
    # 测试
    model.load_state_dict(torch.load(os.path.join(BASE, 'weights', 'best.pth'),
                                     map_location=device, weights_only=True))
    model.eval(); correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            preds = model(data).argmax(1)
            correct += (preds == labels).sum().item(); total += labels.size(0)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    test_acc = correct / total
    print(f"Test Acc: {test_acc:.4f}")

    # 可视化
    # loss / acc 曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], 'b-'); ax1.set_title('Train Loss'); ax1.set_xlabel('Epoch')
    ax2.plot(history['train_acc'], 'b-', label='Train')
    ax2.plot(history['val_acc'], 'r-', label='Val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'loss_acc.png'), dpi=150); plt.close()

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Cat', 'Pred Dog']); ax.set_yticklabels(['True Cat', 'True Dog'])
    ax.set_title(f'Confusion Matrix (Test Acc={test_acc:.4f})'); fig.colorbar(im)
    plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'confusion_matrix.png'), dpi=150); plt.close()
    print(f"Confusion Matrix:\n{cm}")

    # 3. 错误样本（取 8 张拼一张图）
    wrong_idx = np.where(np.array(all_preds) != np.array(all_labels))[0]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, wi in zip(axes.flat, wrong_idx[:8]):
        img, label = test_set[wi]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"true:{'cat' if label==0 else 'dog'} pred:{'cat' if all_preds[wi]==0 else 'dog'}", fontsize=10)
        ax.axis('off')
    plt.suptitle(f'Misclassified Samples ({len(wrong_idx)}/{len(all_labels)})')
    plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'misclassified.png'), dpi=150); plt.close()
    print(f"Misclassified: {len(wrong_idx)}/{len(all_labels)}")

    # 第一层特征图
    img, _ = test_set[0]
    with torch.no_grad():
        feat = model.pool(nn.functional.relu(model.conv1(img.unsqueeze(0).to(device))))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(16):
        axes.flat[i].imshow(feat[0, i].cpu().numpy(), cmap='viridis'); axes.flat[i].axis('off')
    plt.suptitle('First Conv Layer Feature Maps (16 / 32 channels)')
    plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'feature_maps.png'), dpi=150); plt.close()



if __name__ == '__main__':
    main()
