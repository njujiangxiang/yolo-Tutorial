# examples/02-convolution.py
"""
卷积操作演示

理解 2D 卷积的工作原理
"""
import torch
import torch.nn as nn

def main():
    print("=" * 50)
    print("2D 卷积演示")
    print("=" * 50)

    # 创建卷积层
    # input: (batch_size, in_channels, height, width)
    conv = nn.Conv2d(
        in_channels=3,    # 输入通道数（RGB 彩色图）
        out_channels=32,  # 输出通道数（特征图数量）
        kernel_size=3,    # 卷积核大小（3x3）
        stride=1,         # 步长
        padding=1         # 填充（保持尺寸）
    )

    print(f"\n卷积层配置:")
    print(f"  输入通道：3")
    print(f"  输出通道：32")
    print(f"  卷积核：3x3")
    print(f"  步长：1")
    print(f"  填充：1")

    # 创建输入：(batch=1, channels=3, height=224, width=224)
    x = torch.randn(1, 3, 224, 224)
    print(f"\n输入形状：{x.shape}")

    # 前向传播
    out = conv(x)
    print(f"输出形状：{out.shape}")

    # 尺寸计算公式
    print(f"\n输出尺寸计算:")
    print(f"  output_size = (input_size - kernel_size + 2*padding) / stride + 1")
    print(f"  output_size = (224 - 3 + 2*1) / 1 + 1 = 224")

    # 参数数量
    params = sum(p.numel() for p in conv.parameters())
    print(f"\n卷积层参数数量：{params:,}")
    print(f"  计算：(3 * 3 * 3 * 32) + 32 = 896")

    print("\n" + "=" * 50)
    print("关键点:")
    print("=" * 50)
    print("1. 卷积核在图像上滑动，提取局部特征")
    print("2. padding=1 保持输入输出尺寸相同")
    print("3. out_channels 决定输出特征图的数量")
    print("4. 参数共享大大减少了参数数量")

if __name__ == "__main__":
    main()
