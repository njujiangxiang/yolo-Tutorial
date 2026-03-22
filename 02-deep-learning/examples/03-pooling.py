# examples/03-pooling.py
"""
池化层演示

理解最大池化和平均池化的作用
"""
import torch
import torch.nn as nn

def main():
    print("=" * 50)
    print("池化层演示")
    print("=" * 50)

    # 最大池化
    max_pool = nn.MaxPool2d(
        kernel_size=2,  # 池化窗口大小
        stride=2        # 步长
    )

    # 平均池化
    avg_pool = nn.AvgPool2d(
        kernel_size=2,
        stride=2
    )

    print(f"\n池化配置:")
    print(f"  窗口大小：2x2")
    print(f"  步长：2")

    # 创建输入：(batch=1, channels=32, height=100, width=100)
    x = torch.randn(1, 32, 100, 100)
    print(f"\n输入形状：{x.shape}")

    # 最大池化
    out_max = max_pool(x)
    print(f"最大池化输出：{out_max.shape}")

    # 平均池化
    out_avg = avg_pool(x)
    print(f"平均池化输出：{out_avg.shape}")

    # 演示池化效果
    print(f"\n池化作用:")
    print(f"  1. 降维：100x100 → 50x50")
    print(f"  2. 减少计算量：减少 75% 的像素")
    print(f"  3. 防止过拟合")
    print(f"  4. 提取主要特征")

    # 具体示例
    print(f"\n具体示例:")
    small_x = torch.tensor([[[[1., 2., 3., 4.],
                              [5., 6., 7., 8.],
                              [9., 10., 11., 12.],
                              [13., 14., 15., 16.]]]])
    print(f"输入:\n{small_x[0, 0]}")

    small_max = max_pool(small_x)
    print(f"\n最大池化结果:\n{small_max[0, 0]}")

    small_avg = avg_pool(small_x)
    print(f"\n平均池化结果:\n{small_avg[0, 0]}")

if __name__ == "__main__":
    main()
