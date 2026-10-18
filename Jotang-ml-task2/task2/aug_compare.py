"""数据增强对照：去掉所有随机增强，对比"""
import time
import torch
import torch.nn as nn
import numpy as np
from cnn import CatDogCNN, SEED, device
from loader_data import split_files, CatDogDataset, plain_tf

torch.manual_seed(SEED); np.random.seed(SEED)
# 无增强：只 Resize + ToTensor，划分与训练一致
(tr_f, tr_y), (va_f, va_y), _ = split_files()
train_set = CatDogDataset(tr_f, tr_y, plain_tf)
val_set = CatDogDataset(va_f, va_y, plain_tf)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)

# 训练无增强模型
model = CatDogCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
start = time.time()
for epoch in range(5):
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
    print(f"[无增强] Epoch {epoch+1}/5 | Loss {epoch_loss/len(train_set):.4f} | "
          f"Train {train_acc:.4f} | Val {correct/total:.4f}")
print(f"Time: {time.time()-start:.1f}s")
