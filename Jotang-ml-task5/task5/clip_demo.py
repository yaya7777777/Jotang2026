"""任务 2：让 CLIP 配对图片和文字（推理实验）

模型: OpenAI 官方 CLIP RN50（预训练权重，仅推理）
图片: scikit-image 自带样例照片（来源: scikit-image 内置数据集）
流程: 提取特征 -> L2 归一化 -> 余弦相似度矩阵 -> 热力图 -> 干扰描述对比 -> 温度系数实验
"""
import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
import matplotlib.pyplot as plt
import clip
from skimage import data
from PIL import Image

np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = "cuda" if torch.cuda.is_available() else "cpu"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(out, exist_ok=True)

# 加载模型（任务推荐 RN50）
model, preprocess = clip.load("RN50", device=device)# 加载 RN50 模型
print(f"设备: {device}  模型: RN50")

# 准备 4 张内容不同的图片（scikit-image 内置样例，来源已记录）
sources = {
    'astronaut': 'skimage.data.astronaut（NASA 宇航员 Eileen Collins 照片）',
    'coffee': 'skimage.data.coffee（咖啡杯样例照片）',
    'chelsea': 'skimage.data.chelsea（虎斑猫样例照片）',
    'camera': 'skimage.data.camera（经典 cameraman 黑白照片）',
}
imgs = {k: Image.fromarray(getattr(data, k)()) for k in sources}# 加载图片

# 每张图一句简短英文描述（前 4 条正确）+ 故意写错的干扰描述（后 4 条）
texts = [
    "a photo of an astronaut in a spacesuit",        # 图1 正确
    "a photo of two cups of coffee on a table",      # 图2 正确
    "a photo of a tabby cat",                        # 图3 正确
    "a photo of a man taking a picture with a camera",  # 图4 正确
    "a photo of an astronaut swimming in the ocean", # 图1 干扰: 动作/场景错
    "a photo of one single cup of coffee",           # 图2 干扰: 数量错(实际两杯)
    "a photo of a dog",                              # 图3 干扰: 类别错
    "a photo of a man playing a violin",             # 图4 干扰: 动作错
]

# 提取特征并 L2 归一化
with torch.no_grad():
    # 提取图片特征
    img_feat = model.encode_image(torch.stack([preprocess(im) for im in imgs.values()]).to(device))
    # 提取文本特征
    txt_feat = model.encode_text(clip.tokenize(texts).to(device))
print(f"图片特征形状: {img_feat.shape}   文本特征形状: {txt_feat.shape}")

img_feat /= img_feat.norm(dim=-1, keepdim=True)   # L2 归一化 -> 点积即余弦相似度
txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
sim = (img_feat @ txt_feat.T).cpu().numpy()       # (4, 8) 余弦相似度

np.set_printoptions(precision=3, suppress=True, linewidth=150)# 打印相似度矩阵
print("\n4x4 正确配对余弦相似度矩阵:")
print(sim[:, :4])
print("\n每张图最匹配的文字:")
for i, k in enumerate(sources):# 遍历每张图片
    j = int(sim[i].argmax())
    print(f"  {k:<11}-> [{j}] {texts[j]}")

# 干扰描述对比：正确 vs 干扰的相似度差异
print("\n干扰描述对比（关键内容写错后相似度是否下降）:")
for i, k in enumerate(sources):# 遍历每张图片
    d = sim[i, i] - sim[i, 4 + i]
    print(f"  {k:<11} 正确 {sim[i, i]:.3f} vs 干扰 {sim[i, 4+i]:.3f}  (差 {d:+.3f})")

# 热力图：左 4x4 正确配对，右 4x8 含干扰
fig, axes = plt.subplots(1, 2, figsize=(17, 5.2))
short = ['astronaut', 'coffee', 'cat', 'camera']
for ax, mat, title, xlabels in [
        (axes[0], sim[:, :4], '4×4 相似度矩阵（正确描述）', short),
        (axes[1], sim, '4×8 相似度矩阵（含干扰描述）',
         ['正确\nastronaut', '正确\ncoffee', '正确\ncat', '正确\ncamera',
          '干扰\nswimming', '干扰\none cup', '干扰\ndog', '干扰\nviolin'])]:
    im = ax.imshow(mat, cmap='viridis', vmin=0.15, vmax=0.35)
    plt.colorbar(im, ax=ax)
    n_r, n_c = mat.shape
    for r in range(n_r):
        for c in range(n_c):
            ax.text(c, r, f'{mat[r, c]:.2f}', ha='center', va='center', fontsize=7,
                    color='w' if mat[r, c] < 0.28 else 'k')
    ax.set_xticks(range(n_c)); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(short)
    ax.set_title(title); ax.set_xlabel('文字描述'); ax.set_ylabel('图片')
plt.tight_layout()
plt.savefig(os.path.join(out, 'clip_similarity.png'), dpi=150)
plt.close()

# 温度系数实验：同一行相似度乘不同温度后 softmax 的变化
temps = [1/100, 1/14, 1.0]   # CLIP 训练用的 logit_scale≈100(T=0.01) / 论文初始 1/0.07 / 无缩放
row = sim[0]                 # 拿图 0（宇航员）对 8 段文字的相似度
plt.figure(figsize=(8, 5))
xs = range(len(texts))
for T in temps:
    p = np.exp((row - row.max()) / T); p /= p.sum()   # 减最大值防 exp 溢出
    plt.plot(xs, p, 'o-', label=f'T = {T:.3f}')
plt.xticks(xs, [f'[{i}]' for i in range(len(texts))])
plt.ylabel('softmax 概率'); plt.xlabel('文字编号 [0-3]正确 / [4-7]干扰')
plt.title('图0 对 8 段文字: 温度系数对 softmax 分布的影响')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(out, 'temperature.png'), dpi=150)
plt.close()
