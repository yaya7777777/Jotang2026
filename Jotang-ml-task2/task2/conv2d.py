"""实验1：第2题用题目公路图 highway.jpg；手写二维卷积 + 四种核处理猫图 test.jpg"""
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import sys  
# Windows 终端中文防乱码：stdout 强制 UTF-8，并把控制台输出代码页切到 65001
try:
    sys.stdout.reconfigure(encoding='utf-8')
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))

# 手写二维卷积层
def conv2d(img, kernel, padding='valid'):
   
    k_h, k_w = kernel.shape
    if img.ndim == 2:# 灰度图
        h, w = img.shape; c = 1
        img = img[:, :, np.newaxis]# 灰度图转换为通道数为 1 的彩色图
    else:
        h, w, c = img.shape# 彩色图
    if padding == 'same':# 同样填充
        ph, pw = (k_h - 1) // 2, (k_w - 1) // 2# 填充高度和宽度
        img = np.pad(img, ((ph, ph), (pw, pw), (0, 0)))# 对图像进行填充
    out_h, out_w = img.shape[0] - k_h + 1, img.shape[1] - k_w + 1# 输出图像高度和宽度
    kf = kernel.reshape(-1)# 卷积核展开为一维向量
    out = np.zeros((out_h, out_w, c))# 输出图像
    for i in range(out_h):# 遍历输出图像的每个像素
        for j in range(out_w):
            out[i, j] = kf @ img[i:i+k_h, j:j+k_w].reshape(-1, c)# 对每个像素进行卷积操作
    return out.squeeze()# 返回输出图像，去掉通道维度


# Pillow -> NumPy -> Tensor（公路图 highway.jpg）
hw = np.array(Image.open(os.path.join(BASE, 'highway.jpg')).convert('RGB'))# 读取公路图并转换为 RGB 格式
print(f"shape: {hw.shape}, dtype: {hw.dtype}, min: {hw.min()}, max: {hw.max()}")# 打印图像的形状、数据类型、最小值和最大值
print(f"像素 img[270, 480] 的 RGB 值: {hw[270, 480]}（沥青路面，接近灰色）")# 打印像素 (270, 480) 的 RGB 值
print(f"HWC: (H, W, C) = ({hw.shape[0]}, {hw.shape[1]}, {hw.shape[2]})")# 打印图像的 HWC 形状
hw_chw = hw.transpose(2, 0, 1)# 将图像的通道维度移动到第一个位置
print(f"CHW: (C, H, W) = ({hw_chw.shape[0]}, {hw_chw.shape[1]}, {hw_chw.shape[2]})")# 打印图像的 CHW 形状
t = torch.from_numpy(hw_chw)# 将图像转换为 PyTorch Tensor
print(f"Tensor: shape {tuple(t.shape)}, dtype {t.dtype}, min {t.min().item()}, max {t.max().item()}")# 打印 Tensor 的形状、数据类型、最小值和最大值

# 四种卷积核作用于猫图 test.jpg
img = np.array(Image.open(os.path.join(BASE, 'test.jpg')).convert('RGB'))# 读取猫图并转换为 RGB 格式
kernels = {
    '均值模糊 3x3': np.ones((3, 3)) / 9,# 取3x3窗口内的平均值
    '高斯模糊 5x5': np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                              [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256,# 取5x5窗口归一化到 [0, 1] 范围
    '锐化': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),# 5*当前像素-周围4领域像素，放大边缘
    'Sobel-X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),# 水平方向的 Sobel 边缘检测核
    'Sobel-Y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])# 垂直方向的 Sobel 边缘检测核
}

blur = conv2d(img, kernels['均值模糊 3x3'], padding='same')
gauss = conv2d(img, kernels['高斯模糊 5x5'], padding='same')
sharp = conv2d(img, kernels['锐化'], padding='same')
gx = conv2d(img, kernels['Sobel-X'], padding='same')
gy = conv2d(img, kernels['Sobel-Y'], padding='same')
edge = np.sqrt(gx**2 + gy**2).clip(0, 255).astype(np.uint8)# 合并水平和垂直方向的边缘检测结果
valid_out = conv2d(img, kernels['均值模糊 3x3'], padding='valid')# 无填充卷积层

# 可视化结果
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
results = [('原图', img), ('均值模糊', blur.astype(np.uint8)),
           ('高斯模糊', gauss.astype(np.uint8)), ('锐化', sharp.clip(0, 255).astype(np.uint8)),
           ('Sobel-X', gx.clip(0, 255).astype(np.uint8)), ('Sobel-Y', gy.clip(0, 255).astype(np.uint8)),
           ('Sobel 边缘合成', edge), ('valid (无填充)', valid_out.astype(np.uint8))]
for ax, (title, im) in zip(axes.flat, results):
    ax.imshow(im); ax.set_title(title); ax.axis('off')
plt.tight_layout()
os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)
plt.savefig(os.path.join(BASE, 'results', 'conv2d_kernels.png'), dpi=150)
plt.close()

print(f"valid 输出尺寸: {valid_out.shape}, same 输出尺寸: {blur.shape}")

