"""实践 1：简单计算链 f(x, y) = (x*y + x**2)**2

沿计算图手动链式求导，NumPy 反向传播，再用 PyTorch autograd 对照
"""
import sys

import numpy as np
import torch

# Windows 终端中文防乱码：stdout 强制 UTF-8，并把控制台输出代码页切到 65001
try:
    sys.stdout.reconfigure(encoding='utf-8')
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
except Exception:
    pass

# 选定一组 x, y 值，用于计算 f(x, y) 的梯度
x_val, y_val = 2.0, 3.0
print(f"选取 x = {x_val}, y = {y_val}\n")


a = x_val * y_val          # a = x*y
b = x_val ** 2             # b = x^2
c = a + b                  # c = a + b
f = c ** 2                 # f = c^2
print("前向计算:")
print(f"  a = x*y        = {a}")
print(f"  b = x^2        = {b}")
print(f"  c = a+b        = {c}")
print(f"  f = c^2        = {f}")


# NumPy 手动反向传播（链式法则，从后往前）
df = 1.0                       # df/df
dc = df * 2 * c                # f = c^2  -> df/dc = 2c
da = dc * 1.0                  # c = a+b  -> dc/da = 1
db = dc * 1.0                  #          -> dc/db = 1
dx = da * y_val + db * 2 * x_val   # a=x*y -> da/dx=y ; b=x^2 -> db/dx=2x
dy = da * x_val                    # a=x*y -> da/dy=x

print("\nNumPy 手动反向传播梯度:")
print(f"  df/dx = {dx}")
print(f"  df/dy = {dy}")

# 理论解析解验证: f = (xy+x^2)^2
# df/dx = 2(xy+x^2)(y+2x), df/dy = 2(xy+x^2)x
print("\n解析解:")
print(f"  df/dx = {2*c*(y_val + 2*x_val)}")
print(f"  df/dy = {2*c*x_val}")

# PyTorch autograd 对照
tx = torch.tensor(x_val, dtype=torch.float64, requires_grad=True)
ty = torch.tensor(y_val, dtype=torch.float64, requires_grad=True)
tf = (tx * ty + tx ** 2) ** 2
tf.backward()

print("\nPyTorch autograd 梯度:")
print(f"  df/dx = {tx.grad.item()}")
print(f"  df/dy = {ty.grad.item()}")


# 逐项比较
err_x = abs(dx - tx.grad.item())# df/dx 误差
err_y = abs(dy - ty.grad.item())# df/dy 误差
print("\n对照结果")
print(f"df/dx: NumPy={dx:.6f}  PyTorch={tx.grad.item():.6f}  误差={err_x:.2e}")
print(f"df/dy: NumPy={dy:.6f}  PyTorch={ty.grad.item():.6f}  误差={err_y:.2e}")
print(f"最大绝对误差 = {max(err_x, err_y):.2e}")# 最大绝对误差
print("结果一致!" if max(err_x, err_y) < 1e-9 else "结果不一致!")
