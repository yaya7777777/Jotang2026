"""实验2：猫狗分类 CNN（精简版）"""
import os, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei']

# 数据（用本地图片生成假数据跑通，Kaggle 猫狗数据集需要下载后替换）
from torchvision import transforms
# 生成假数据：猫=红色块，狗=蓝色块
def make_fake_data(n=200, size=64):
    imgs, labels = [], []
    for i in range(n):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        if i % 2 == 0:
            img[:, :] = [200, 50, 50]  # 猫=红
            labels.append(0)
        else:
            img[:, :] = [50, 50, 200]  # 狗=蓝
            labels.append(1)
        # 加噪声
        img = img + np.random.randint(-20, 20, img.shape, dtype=np.int16).astype(np.uint8)
        imgs.append(img)
    return np.array(imgs), np.array(labels)

train_imgs, train_labels = make_fake_data(800)
test_imgs, test_labels = make_fake_data(200)

class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

train_set = FakeDataset(train_imgs, train_labels, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
test_set = FakeDataset(test_imgs, test_labels, transform=transforms.ToTensor())
n_train = int(0.8 * len(train_set)); n_val = len(train_set) - n_train
train_set, val_set = torch.utils.data.random_split(train_set, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

# CNN
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))  # 64->32
        x = self.pool(nn.ReLU()(self.conv2(x)))  # 32->16
        x = self.pool(nn.ReLU()(self.conv3(x)))  # 16->8
        x = x.view(-1, 128 * 8 * 8)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)

os.makedirs('results', exist_ok=True)
os.makedirs('weights', exist_ok=True)

model = CatDogCNN().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# 记录形状
x = torch.randn(1, 3, 64, 64).to(device)
for name, layer in [('conv1', model.conv1), ('pool1', model.pool), ('conv2', model.conv2),
                    ('pool2', model.pool), ('conv3', model.conv3), ('pool3', model.pool)]:
    x = layer(x)
    print(f"{name}: {x.shape}")

# 训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
best_acc = 0
start = time.time()
for epoch in range(30):
    model.train(); correct = 0; total = 0; epoch_loss = 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        epoch_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item(); total += labels.size(0)
    train_acc = correct / total
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            correct += (outputs.argmax(1) == labels).sum().item(); total += labels.size(0)
    val_acc = correct / total
    history['train_loss'].append(epoch_loss / total)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'weights/best.pth')
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30 | Loss {epoch_loss/total:.4f} | Train {train_acc:.4f} | Val {val_acc:.4f}")
print(f"Time: {time.time()-start:.2f}s | Best Val: {best_acc:.4f}")

# 测试
model.load_state_dict(torch.load('weights/best.pth', map_location=device, weights_only=True))
model.eval(); correct = 0; total = 0
all_preds, all_labels = [], []
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item(); total += labels.size(0)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
print(f"Test Acc: {correct/total:.4f}")

# loss/acc 曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], 'b-', label='Train Loss'); ax1.legend(); ax1.set_title('Loss')
ax2.plot(history['train_acc'], 'b-', label='Train Acc')
ax2.plot(history['val_acc'], 'r-', label='Val Acc')
ax2.legend(); ax2.set_title('Accuracy')
plt.tight_layout(); plt.savefig('results/loss_acc.png', dpi=150); plt.close()

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
ax.set_title('Confusion Matrix'); fig.colorbar(im)
plt.tight_layout(); plt.savefig('results/confusion_matrix.png', dpi=150); plt.close()

# 错误样本
wrong = np.array(all_preds) != np.array(all_labels)
print(f"Misclassified: {wrong.sum()}/{len(all_labels)}")

# 特征图可视化
model.eval()
test_img, test_label = next(iter(test_loader))
test_img = test_img.to(device)
x = model.pool(nn.ReLU()(model.conv1(test_img)))  # 第一层特征图
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    axes.flat[i].imshow(x[0, i].cpu().detach().numpy(), cmap='viridis')
    axes.flat[i].axis('off')
plt.suptitle('First Conv Layer Feature Maps (16 channels)')
plt.tight_layout(); plt.savefig('results/feature_maps.png', dpi=150); plt.close()

# 数据增强对比
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
orig_img, _ = test_set[0]
aug = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding=8),
                         transforms.ColorJitter(brightness=0.3, contrast=0.3), transforms.RandomRotation(15)])
for i in range(8):
    axes.flat[i].imshow(np.array(aug(transforms.ToPILImage()(orig_img)).convert('RGB')))
    axes.flat[i].axis('off')
plt.suptitle('Data Augmentation Examples')
plt.tight_layout(); plt.savefig('results/augmentation.png', dpi=150); plt.close()

print("done")
