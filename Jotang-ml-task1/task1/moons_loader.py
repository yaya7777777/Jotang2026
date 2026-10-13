from sklearn.datasets import make_moons
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MoonsDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.15, random_state=42):
        # 直接调用 make_moons 生成数据
        data, labels = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
        )


        # 保存到实例变量
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
