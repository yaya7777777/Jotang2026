"""拓展 2/3：首末层特征图 L2 范数对比+BasicBlock vs Bottleneck 参数量/速度"""
import os, time
import torch
from PIL import Image
import torchvision.transforms as T
from cnn import CatDogCNN
from resnet import ResNet, BasicBlock, Bottleneck
import sys
# Windows 终端中文防乱码：stdout 强制 UTF-8，并把控制台输出代码页切到 65001
try:
    sys.stdout.reconfigure(encoding='utf-8')
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 首层 vs 末层特征图 L2 范数
model = CatDogCNN().to(device)
model.load_state_dict(torch.load(os.path.join(BASE, 'weights', 'best.pth'), map_location=device))
model.eval()
tf = T.Compose([T.Resize((64, 64)), T.ToTensor()])
x = tf(Image.open(os.path.join(BASE, 'test.jpg')).convert('RGB')).unsqueeze(0).to(device)

with torch.no_grad():
    x1 = model.pool(torch.relu(model.conv1(x)))      # 首层 1×32×32×32
    x2 = model.pool(torch.relu(model.conv2(x1)))
    x3 = model.pool(torch.relu(model.conv3(x2)))    # 末层 1×128×8×8
# b×c×h×w 沿 h,w 求 L2 范数 -> b×c
l2_first = x1.norm(p=2, dim=(2, 3)).squeeze(0).cpu().numpy()# 
l2_last = x3.norm(p=2, dim=(2, 3)).squeeze(0).cpu().numpy()
print("首层 vs 末层特征图 L2 范数（沿 H,W）")
print(f"首层 conv1 (32通道): mean={l2_first.mean():.3f} std={l2_first.std():.3f} "
      f"min={l2_first.min():.3f} max={l2_first.max():.3f}")
print(f"末层 conv3 (128通道): mean={l2_last.mean():.3f} std={l2_last.std():.3f} "
      f"min={l2_last.min():.3f} max={l2_last.max():.3f}")
print(f"范数<0.1 的静默通道: 首层 {(l2_first < 0.1).sum()}/32, 末层 {(l2_last < 0.1).sum()}/128")

# BasicBlock vs Bottleneck 参数量/速度
def count_params(m):
    return sum(p.numel() for p in m.parameters())

def bench(m, x, n=30):
    m.eval()
    with torch.no_grad():
        for _ in range(5): m(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            for _ in range(n): m(x)
        if device.type == 'cuda': torch.cuda.synchronize()
    return (time.time() - t) / n * 1000

m18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2).to(device)
m50 = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2).to(device)
xb = torch.randn(32, 3, 224, 224, device=device)
print("\nBasicBlock(ResNet-18) vs Bottleneck(ResNet-50)")
print(f"ResNet-18 (BasicBlock): 参数 {count_params(m18):,}, batch=32 前向 {bench(m18, xb):.1f} ms")
print(f"ResNet-50 (Bottleneck): 参数 {count_params(m50):,}, batch=32 前向 {bench(m50, xb):.1f} ms")

