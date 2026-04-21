import torch
# 如果你需要加载你的 UFD detector，请在这里先导包和初始化
# from your_model_file import build_detector_from_ufd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 第 3 件事：检查 checkpoint 存的是什么 ---
# path = "你的权重文件路径.pth"
# obj = torch.load(path, map_location="cpu")
# print("Checkpoint type:", type(obj))

# 假设你已经构建好了 detector (请用你真实的加载代码替换下面这行)
# detector = build_detector_from_ufd(...)
# detector.to(device)

# 为了演示代码不报错，这里临时用一个空的 nn.Linear 模拟你的 detector
import torch.nn as nn
detector = nn.Linear(256 * 256 * 3, 1).to(device) 

# --- 第 1 件事：验证梯度通不通 ---
print("--- 开始验证梯度 ---")
# 注意：如果你用的是真实的图像模型，输入形状通常是 (B, C, H, W)
x = torch.rand(1, 3, 256, 256, device=device, requires_grad=True)

# 前向传播 (如果是上面的模拟 Linear，需要把 x 展平)
# 这里保留你原本的写法：
# score = detector(x).mean() 

# 模拟的前向传播
score = detector(x.view(1, -1)).mean() 

score.backward()

if x.grad is not None:
    print("梯度绝对值平均:", x.grad.abs().mean().item())
    print("✅ 梯度正常 (非空且非全零)")
else:
    print("❌ 梯度为空！Wrapper 可能存在截断梯度的操作 (如 torch.no_grad() 或 .detach())")

# --- 第 2 件事：UFD score 的方向 ---
# 这个需要你传入一张真实的 Fake 图像和一张真实的 Real 图像来 print(score) 观察

