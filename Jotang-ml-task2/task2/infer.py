"""推理程序：输入一张本地图片，输出猫/狗及预测概率"""
import os, argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
# Windows 终端中文防乱码：stdout 强制 UTF-8，并把控制台输出代码页切到 65001
try:
    sys.stdout.reconfigure(encoding='utf-8')
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass

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
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

BASE = os.path.dirname(os.path.abspath(__file__))

def predict(img_path, model_path=None):
    model_path = model_path or os.path.join(BASE, 'weights', 'best.pth')
    # 相对路径：优先按当前目录找，找不到再按脚本所在目录找，两种 cwd 下都能跑
    if not os.path.isabs(img_path) and not os.path.exists(img_path):
        img_path = os.path.normpath(os.path.join(BASE, img_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    # 与训练/测试时一致的预处理
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    x = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    cat_p, dog_p = probs[0].item(), probs[1].item()
    label = '猫' if cat_p > dog_p else '狗'
    print(f"{label}  (cat {cat_p*100:.1f}% / dog {dog_p*100:.1f}%)")
    return label, cat_p, dog_p

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default=os.path.join('..', 'test1', '1.jpg'),
                        help='输入图片路径（默认 ../test1/1.jpg）')
    parser.add_argument('--model', default=None, help='模型权重路径（默认 weights/best.pth）')
    args = parser.parse_args()
    predict(args.img, args.model)
