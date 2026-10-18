"""加载训练好的模型，在测试集上评估并生成全部结果图"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
# 数据读取与预处理全部来自 loader_data.py（同一份划分，保证与训练一致）
from loader_data import (SEED, BASE, eval_tf, train_tf, split_files,
                         CatDogDataset, is_placeholder)

np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False

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
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc2(nn.functional.relu(self.fc1(x)))

# 与训练时完全一致的划分（来自 loader_data.py）
(_, _), (_, _), (te_f, te_y) = split_files()
test_set = CatDogDataset(te_f, te_y, eval_tf)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, num_workers=0)
print(f"Test: {len(test_set)}")

model = CatDogCNN().to(device)
model.load_state_dict(torch.load(os.path.join(BASE, 'weights', 'best.pth'),
                                 map_location=device, weights_only=True))
model.eval()
correct = total = 0
all_preds, all_labels = [], []
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        preds = model(data).argmax(1)
        correct += (preds == labels).sum().item(); total += labels.size(0)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
test_acc = correct / total
print(f"Test Acc: {test_acc:.4f}")

# 1. 混淆矩阵
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
tn, fp, fn, tp = cm.ravel()
print(f"Cat: Precision {tn/(tn+fn):.4f} Recall {tn/(tn+fp):.4f}")
print(f"Dog: Precision {tp/(tp+fp):.4f} Recall {tp/(tp+fn):.4f}")

# 2. 错误样本
wrong_idx = np.where(np.array(all_preds) != np.array(all_labels))[0]
wrong_show = [wi for wi in wrong_idx if not is_placeholder(test_set.files[wi])]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, wi in zip(axes.flat, wrong_show[:8]):
    img, label = test_set[wi]
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(f"true:{'cat' if label==0 else 'dog'} pred:{'cat' if all_preds[wi]==0 else 'dog'}", fontsize=10)
    ax.axis('off')
plt.suptitle(f'Misclassified Samples ({len(wrong_idx)}/{len(all_labels)})')
plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'misclassified.png'), dpi=150); plt.close()
print(f"Misclassified: {len(wrong_idx)}/{len(all_labels)}")

# 3. 第一层特征图（16/32 通道）
img, _ = test_set[0]
with torch.no_grad():
    feat = model.pool(nn.functional.relu(model.conv1(img.unsqueeze(0).to(device))))
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    axes.flat[i].imshow(feat[0, i].cpu().numpy(), cmap='viridis'); axes.flat[i].axis('off')
plt.suptitle('First Conv Layer Feature Maps (16 / 32 channels)')
plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'feature_maps.png'), dpi=150); plt.close()

# 4. 最后一层卷积特征图（16/128 通道，8x8）
with torch.no_grad():
    x = model.pool(nn.functional.relu(model.conv1(img.unsqueeze(0).to(device))))
    x = model.pool(nn.functional.relu(model.conv2(x)))
    x = model.pool(nn.functional.relu(model.conv3(x)))
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    axes.flat[i].imshow(x[0, i].cpu().numpy(), cmap='viridis'); axes.flat[i].axis('off')
plt.suptitle('Last Conv Layer Feature Maps (16 / 128 channels, 8x8)')
plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'feature_maps_last.png'), dpi=150); plt.close()

#   数据增强前后对比
#   上排：原图 + 三种单一增强（只做一种变换，对比每种操作的单独效果）
#   下排：完整 pipeline 随机采样 × 4（展示训练时实际喂给模型的多样化样本）
pil_img = Image.open(test_set.files[0]).convert('RGB')
# 单一变换（各自只改变一个维度，方便对比）
resize_tf  = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
flip_tf    = transforms.Compose([transforms.Resize((64, 64)),
                                  transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()])
crop_tf    = transforms.Compose([transforms.Resize((64, 64)),
                                  transforms.RandomCrop(64, padding=4), transforms.ToTensor()])
color_tf   = transforms.Compose([transforms.Resize((64, 64)),
                                  transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                  transforms.ToTensor()])

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
# 上排：原图 + 三种单一增强
axes[0, 0].imshow(resize_tf(pil_img).permute(1, 2, 0).numpy())
axes[0, 0].set_title('原图 Resize(64,64)', fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(flip_tf(pil_img).permute(1, 2, 0).numpy())
axes[0, 1].set_title('仅水平翻转\n(左右镜像)', fontsize=10)
axes[0, 1].axis('off')

axes[0, 2].imshow(crop_tf(pil_img).permute(1, 2, 0).numpy())
axes[0, 2].set_title('仅随机裁剪\n(padding=4, 平移±4px)', fontsize=10)
axes[0, 2].axis('off')

axes[0, 3].imshow(color_tf(pil_img).permute(1, 2, 0).numpy())
axes[0, 3].set_title('仅颜色扰动\n(亮度/对比度 ±20%)', fontsize=10)
axes[0, 3].axis('off')

# 下排：完整 pipeline（水平翻转 + 随机裁剪 + 颜色扰动）随机采样 × 4
for i in range(4):
    axes[1, i].imshow(train_tf(pil_img).permute(1, 2, 0).numpy())
    axes[1, i].set_title(f'完整增强 #{i+1}\n(翻转+裁剪+颜色)', fontsize=10)
    axes[1, i].axis('off')

plt.suptitle('数据增强效果对比（同一张图，不同变换组合）', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'augmentation.png'), dpi=150); plt.close()

# 数据集随机样本展示
rng = np.random.RandomState(0)
clean_pool = [i for i in range(len(test_set)) if not is_placeholder(test_set.files[i])]
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, i in zip(axes.flat, rng.choice(clean_pool, 10, replace=False)):
    img, label = test_set[i]
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title('cat' if label == 0 else 'dog'); ax.axis('off')
plt.suptitle('Random Samples from Dataset')
plt.tight_layout(); plt.savefig(os.path.join(BASE, 'results', 'dataset_samples.png'), dpi=150); plt.close()

