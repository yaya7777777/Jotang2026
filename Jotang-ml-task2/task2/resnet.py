"""手写 ResNet 主干（BasicBlock / Bottleneck 可替换）"""
import torch
import torch.nn as nn

# 基本块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)# 归一化层
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()#  shortcut 连接层
        if stride != 1 or inplanes != planes:# 当步长不是 1 或输入通道数不是输出通道数时，需要使用 shortcut 连接
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    """1x1 降维 -> 3x3 -> 1x1 升维，ResNet-50/101/152 使用"""
    expansion = 4# 升维比例

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)# 升维卷积层
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.shortcut = nn.Sequential()#  shortcut 连接层
        if stride != 1 or inplanes != planes * 4:# 当步长不是 1 或输入通道数不是输出通道数的 4 倍时，需要使用 shortcut 连接
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, 1, stride, bias=False),
                nn.BatchNorm2d(planes * 4))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return torch.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)# 全局平均池化层
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # 堆叠块
    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion# 更新输入通道数
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def ResNet18(num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


if __name__ == '__main__':
    model = ResNet18(num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    print(f"Input: {x.shape}")
    print(f"Output: {model(x).shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
