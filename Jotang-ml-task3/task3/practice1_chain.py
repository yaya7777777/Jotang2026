"""实践 1：简单计算链 f(x, y) = (x*y + x**2)**2

沿计算图手动链式求导，NumPy 反向传播，再用 PyTorch autograd 对照。
"""
import numpy as np
import torch

# ============================================================
# 选定一组 x, y
# ============================================================
x_val, y_val = 2.0, 3.0
print(f"选取 x = {x_val}, y = {y_val}\n")

# ============================================================
# 1. 计算图拆解（前向）
#
#   a = x * y        (乘法)
#   b = x ** 2       (平方)
#   c = a + b        (加法)
#   f = c ** 2       (平方)
#
#   x ──┬──> [*]──> a ──┐
#       │               │
#       └──> [^2]─> b ──┴──> [+]──> c ──> [^2]──> f
#   y ───────────────^
# ============================================================
a = x_val * y_val          # a = x*y
b = x_val ** 2             # b = x^2
c = a + b                  # c = a + b
f = c ** 2                 # f = c^2
print("前向计算:")
print(f"  a = x*y        = {a}")
print(f"  b = x^2        = {b}")
print(f"  c = a+b        = {c}")
print(f"  f = c^2        = {f}")

# ============================================================
# 2. NumPy 手动反向传播（链式法则，从后往前）
#
#   df/df = 1
#   df/dc = 2c                    (f = c^2)
#   df/da = df/dc * dc/da = 2c*1  (c = a+b)
#   df/db = df/dc * dc/db = 2c*1
#   df/dx = df/da*da/dx + df/db*db/dx
#         = 2c*y + 2c*2x
#   df/dy = df/da*da/dy = 2c*x
# ============================================================
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

# ============================================================
# 3. PyTorch autograd 对照
# ============================================================
tx = torch.tensor(x_val, dtype=torch.float64, requires_grad=True)
ty = torch.tensor(y_val, dtype=torch.float64, requires_grad=True)
tf = (tx * ty + tx ** 2) ** 2
tf.backward()

print("\nPyTorch autograd 梯度:")
print(f"  df/dx = {tx.grad.item()}")
print(f"  df/dy = {ty.grad.item()}")

# ============================================================
# 4. 逐项比较
# ============================================================
err_x = abs(dx - tx.grad.item())
err_y = abs(dy - ty.grad.item())
print("\n=== 对照结果 ===")
print(f"df/dx: NumPy={dx:.6f}  PyTorch={tx.grad.item():.6f}  误差={err_x:.2e}")
print(f"df/dy: NumPy={dy:.6f}  PyTorch={ty.grad.item():.6f}  误差={err_y:.2e}")
print(f"最大绝对误差 = {max(err_x, err_y):.2e}")
print("结果一致!" if max(err_x, err_y) < 1e-9 else "结果不一致!")
