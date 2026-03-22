# examples/06-optimizers.py
"""
优化器演示

理解常用优化器的特点和使用
"""
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    print("=" * 50)
    print("优化器演示")
    print("=" * 50)

    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

    # 1. SGD
    sgd = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,      # 动量，加速收敛
        weight_decay=0.0001  # L2 正则化
    )
    print("\n1. SGD (随机梯度下降)")
    print("   参数:")
    print("   - lr: 学习率 (0.01)")
    print("   - momentum: 动量 (0.9)")
    print("   - weight_decay: L2 正则化 (0.0001)")
    print("   特点：经典，需要调参")

    # 2. Adam
    adam = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),  # 动量参数
        eps=1e-08,           # 数值稳定性
        weight_decay=0.0     # 默认无权重衰减
    )
    print("\n2. Adam")
    print("   参数:")
    print("   - lr: 学习率 (0.001)")
    print("   - betas: 动量参数 (0.9, 0.999)")
    print("   特点：自适应学习率，推荐默认")

    # 3. AdamW
    adamw = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01  # 更好的权重衰减
    )
    print("\n3. AdamW")
    print("   特点：Adam 改进版，权重衰减更好")

    # 4. RMSprop
    rmsprop = optim.RMSprop(
        model.parameters(),
        lr=0.01,
        alpha=0.99  # 平滑系数
    )
    print("\n4. RMSprop")
    print("   特点：适合 RNN")

    print("\n" + "=" * 50)
    print("选择指南:")
    print("=" * 50)
    print("默认 → Adam (lr=0.001)")
    print("需要精细调参 → SGD+ 动量")
    print("训练不稳定 → AdamW")
    print("RNN → RMSprop")

if __name__ == "__main__":
    main()
