"""推理程序：输入本地图片，输出猫/狗预测概率"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import argparse

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
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)

def predict(img_path, model_path='weights/best.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    img = Image.open(img_path).convert('RGB').resize((64, 64))
    x = torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    cat_prob = probs[0, 0].item()
    dog_prob = probs[0, 1].item()
    label = '猫' if cat_prob > dog_prob else '狗'
    print(f"{label} (猫 {cat_prob*100:.1f}% / 狗 {dog_prob*100:.1f}%)")
    return img, label, cat_prob, dog_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='输入图片路径')
    parser.add_argument('--model', default='weights/best.pth', help='模型权重路径')
    args = parser.parse_args()
    predict(args.img, args.model)
