# examples/01-activation-functions.py
"""
激活函数演示

理解常用激活函数的作用和特点
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 创建输入数据
x = torch.linspace(-5, 5, 100)

# 激活函数
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=0)

# 计算输出
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_softmax = softmax(x)

print("激活函数说明:")
print("=" * 50)
print("ReLU: f(x) = max(0, x)")
print("  - 最常用，计算快")
print("  - 梯度不消失")
print("  - 用于隐藏层")
print()
print("Sigmoid: f(x) = 1 / (1 + exp(-x))")
print("  - 输出范围：0-1")
print("  - 用于二分类输出层")
print("  - 容易梯度消失")
print()
print("Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))")
print("  - 输出范围：-1 到 1")
print("  - 零中心化")
print()
print("Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))")
print("  - 输出概率分布")
print("  - 用于多分类输出层")

if __name__ == "__main__":
    # 如果安装了 matplotlib，可以绘图
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        x_np = x.numpy()
        axes[0, 0].plot(x_np, y_relu.numpy())
        axes[0, 0].set_title('ReLU')
        axes[0, 0].grid(True)

        axes[0, 1].plot(x_np, y_sigmoid.numpy())
        axes[0, 1].set_title('Sigmoid')
        axes[0, 1].grid(True)

        axes[1, 0].plot(x_np, y_tanh.numpy())
        axes[1, 0].set_title('Tanh')
        axes[1, 0].grid(True)

        axes[1, 1].plot(x_np, y_softmax.numpy())
        axes[1, 1].set_title('Softmax')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('activation_functions.png')
        print("\n图像已保存：activation_functions.png")
    except ImportError:
        print("\n未安装 matplotlib，跳过绘图")
