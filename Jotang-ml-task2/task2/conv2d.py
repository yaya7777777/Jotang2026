"""实验1：手写二维卷积 + 四种核处理同一张图"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def conv2d(img, kernel, padding='valid'):
    """手写 2D 卷积（不用现成实现）"""
    k_h, k_w = kernel.shape
    if img.ndim == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape; c = 1; img = img[:, :, np.newaxis]
    if padding == 'same':
        ph, pw = (k_h - 1) // 2, (k_w - 1) // 2
        img = np.pad(img, ((ph, ph), (pw, pw), (0, 0)))
    out_h, out_w = img.shape[0] - k_h + 1, img.shape[1] - k_w + 1
    out = np.zeros((out_h, out_w, c))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = (img[i:i+k_h, j:j+k_w] * kernel[:, :, np.newaxis]).sum(axis=(0, 1))
    return out.squeeze()

import os
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.jpg')
img = np.array(Image.open(img_path).convert('RGB'))
print(f"shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
print(f"HWC: (H, W, C) = ({img.shape[0]}, {img.shape[1]}, {img.shape[2]})")
img_chw = img.transpose(2, 0, 1)
print(f"CHW: (C, H, W) = ({img_chw.shape[0]}, {img_chw.shape[1]}, {img_chw.shape[2]})")

kernels = {
    '均值模糊 3x3': np.ones((3, 3)) / 9,
    '高斯模糊 5x5': np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                              [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256,
    '锐化': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'Sobel-X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'Sobel-Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
}

# Sobel 合成边缘图
gx = conv2d(img, kernels['Sobel-X'], padding='same')
gy = conv2d(img, kernels['Sobel-Y'], padding='same')
edge = np.sqrt(gx**2 + gy**2).clip(0, 255).astype(np.uint8)

# valid vs same 对比
valid_out = conv2d(img, kernels['均值模糊 3x3'], padding='valid')
same_out = conv2d(img, kernels['均值模糊 3x3'], padding='same')

# 画图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0, 0].imshow(img); axes[0, 0].set_title('原图')
axes[0, 1].imshow(conv2d(img, kernels['均值模糊 3x3'], padding='same').astype(np.uint8)); axes[0, 1].set_title('均值模糊')
axes[0, 2].imshow(conv2d(img, kernels['高斯模糊 5x5'], padding='same').astype(np.uint8)); axes[0, 2].set_title('高斯模糊')
axes[0, 3].imshow(conv2d(img, kernels['锐化'], padding='same').clip(0, 255).astype(np.uint8)); axes[0, 3].set_title('锐化')
axes[1, 0].imshow(gx.clip(0, 255).astype(np.uint8)); axes[1, 0].set_title('Sobel-X')
axes[1, 1].imshow(gy.clip(0, 255).astype(np.uint8)); axes[1, 1].set_title('Sobel-Y')
axes[1, 2].imshow(edge); axes[1, 2].set_title('Sobel 边缘合成')
axes[1, 3].imshow(valid_out.astype(np.uint8)); axes[1, 3].set_title('valid (无填充)')
for ax in axes.flat: ax.axis('off')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/conv2d_kernels.png', dpi=150); plt.close()

print(f"valid 输出尺寸: {valid_out.shape}, same 输出尺寸: {same_out.shape}")
print("done")
