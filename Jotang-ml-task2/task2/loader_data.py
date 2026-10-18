"""数据读取与预处理：数据集类、预处理变换、8:1:1 划分、DataLoader"""
import os, sys, hashlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Windows 终端中文防乱码：stdout 强制 UTF-8，并把控制台输出代码页切到 65001
try:
    sys.stdout.reconfigure(encoding='utf-8')
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass

SEED = 42
plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE, '..', 'train')
RESULTS_DIR = os.path.join(BASE, 'results')

# Kaggle 数据集里的损坏占位图 "No Photo Available"，
# 不是真实猫狗照片，展示样本时应剔除
PLACEHOLDER_MD5 = '261d939bb330283fae86b541e25576c8'

def is_placeholder(path):# 检查文件是否为占位图
    return hashlib.md5(open(path, 'rb').read()).hexdigest() == PLACEHOLDER_MD5

# 预处理
# 训练集：随机水平翻转 + 随机裁剪 + 颜色扰动（数据增强）
train_tf = transforms.Compose([
    transforms.Resize((64, 64)),# 缩放为 64x64
    transforms.RandomHorizontalFlip(),# 随机水平翻转
    transforms.RandomCrop(64, padding=4),# 随机裁剪 64x64
    transforms.ColorJitter(brightness=0.2, contrast=0.2),# 颜色扰动
    transforms.ToTensor()])
# 验证/测试集：不进行数据增强
plain_tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])# 缩放为 64x64 并转换为张量
eval_tf = plain_tf

# 数据集类
class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels, tf):
        self.files, self.labels, self.tf = files, labels, tf
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')# 读取并转换为 RGB 格式
        return self.tf(img), self.labels[i]


# 划分数据集
def split_files():
    names = sorted(os.listdir(TRAIN_DIR))                       # 排序保证跨机器可复现
    files = [os.path.join(TRAIN_DIR, n) for n in names]
    labels = [0 if n.startswith('cat') else 1 for n in names]
    idx = np.random.RandomState(SEED).permutation(len(files))# 随机打乱索引
    n1, n2 = int(0.8 * len(idx)), int(0.9 * len(idx))
    def pick(sl):# 从索引切片中提取文件路径和标签
        return [files[i] for i in idx[sl]], [labels[i] for i in idx[sl]]
    return pick(slice(0, n1)), pick(slice(n1, n2)), pick(slice(n2, None))

# 构建数据集
def build_datasets(train_transform=None, eval_transform=None):
    (tr_f, tr_y), (va_f, va_y), (te_f, te_y) = split_files()
    train_transform = train_tf if train_transform is None else train_transform# 训练集使用 train_tf
    eval_transform = eval_tf if eval_transform is None else eval_transform# 验证/测试集使用 eval_tf
    return (CatDogDataset(tr_f, tr_y, train_transform),
            CatDogDataset(va_f, va_y, eval_transform),
            CatDogDataset(te_f, te_y, eval_transform))


def build_loaders(batch_size=64, num_workers=0, train_transform=None, eval_transform=None):
    
    train_set, val_set, test_set = build_datasets(train_transform, eval_transform)
    DL = torch.utils.data.DataLoader# 构建 DataLoader 类
    return (DL(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),# 组成batch
            DL(val_set, batch_size=128, num_workers=num_workers),
            DL(test_set, batch_size=128, num_workers=num_workers),
            train_set, val_set, test_set)


# batch 组成 + 预处理前后对比
def show_batch_and_compare(n_show=6):
    # 1) 一个 batch 是怎么组成的
    train_loader, _, _, train_set, val_set, test_set = build_loaders(batch_size=64)
    images, labels = next(iter(train_loader))
    print(f"一个 batch: images {tuple(images.shape)} {images.dtype}, 值域 [{images.min():.2f}, {images.max():.2f}]")
    print(f"           labels {tuple(labels.shape)} {labels.dtype}  (0=cat, 1=dog)")
    print(f"数据集张数: Train {len(train_set)} | Val {len(val_set)} | Test {len(test_set)}")

    # 2) 预处理前后随机对比（同一张图，上排原图 / 下排预处理后）
    rng = np.random.RandomState(SEED)
    pool = [n for n in sorted(os.listdir(TRAIN_DIR)) if not is_placeholder(os.path.join(TRAIN_DIR, n))]# 筛选出非占位图的文件名
    picks = rng.choice(pool, n_show, replace=False)# 随机选择 n_show 张图

    fig, axes = plt.subplots(2, n_show, figsize=(3.0 * n_show, 6.5))
    for col, name in enumerate(picks):
        path = os.path.join(TRAIN_DIR, name)
        pil = Image.open(path).convert('RGB')
        w, h = pil.size
        # 上排：原图（保持原始尺寸/纵横比）
        axes[0, col].imshow(np.array(pil))
        axes[0, col].set_title(f"原图 {w}×{h}  ({name.split('.')[0]})", fontsize=9)
        axes[0, col].axis('off')
        # 下排：同一张图经预处理后（64×64 张量，还原成 HWC 显示）
        tensor = eval_tf(pil)
        axes[1, col].imshow(tensor.permute(1, 2, 0).numpy())
        axes[1, col].set_title(f"预处理后 {tuple(tensor.shape)}", fontsize=9)
        axes[1, col].axis('off')
        print(f"{name}: 原图 {w}×{h} (纵横比 {w/h:.2f}) → 预处理后 64×64 (纵横比 1.00)")

    plt.suptitle('Preprocess Before / After (Resize 64×64 + ToTensor)')
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'preprocess_compare.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    (tr_f, _), (va_f, _), (te_f, _) = split_files()
    print(f"划分结果: Train {len(tr_f)} | Val {len(va_f)} | Test {len(te_f)}")
    show_batch_and_compare()
