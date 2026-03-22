# examples/05-loss-functions.py
"""
损失函数演示

理解常用损失函数的使用场景
"""
import torch
import torch.nn as nn

def main():
    print("=" * 50)
    print("损失函数演示")
    print("=" * 50)

    # 1. 交叉熵损失（多分类）
    print("\n1. 交叉熵损失 (CrossEntropyLoss)")
    print("   用途：多分类问题")
    ce_loss = nn.CrossEntropyLoss()
    predictions = torch.randn(3, 5)  # 3 个样本，5 个类别
    targets = torch.tensor([1, 0, 4])  # 真实标签
    loss = ce_loss(predictions, targets)
    print(f"   预测形状：{predictions.shape}")
    print(f"   标签形状：{targets.shape}")
    print(f"   损失值：{loss.item():.4f}")

    # 2. 均方误差（回归）
    print("\n2. 均方误差 (MSELoss)")
    print("   用途：回归问题")
    mse_loss = nn.MSELoss()
    predictions = torch.randn(3, 1)
    targets = torch.randn(3, 1)
    loss = mse_loss(predictions, targets)
    print(f"   损失值：{loss.item():.4f}")

    # 3. 二元交叉熵（二分类）
    print("\n3. 二元交叉熵 (BCELoss)")
    print("   用途：二分类问题")
    bce_loss = nn.BCELoss()
    predictions = torch.sigmoid(torch.randn(3, 1))
    targets = torch.rand(3, 1)
    loss = bce_loss(predictions, targets)
    print(f"   损失值：{loss.item():.4f}")

    # 4. L1 损失
    print("\n4. L1 损失 (L1Loss)")
    print("   用途：回归，对异常值鲁棒")
    l1_loss = nn.L1Loss()
    predictions = torch.randn(3, 1)
    targets = torch.randn(3, 1)
    loss = l1_loss(predictions, targets)
    print(f"   损失值：{loss.item():.4f}")

    print("\n" + "=" * 50)
    print("选择指南:")
    print("=" * 50)
    print("多分类 → CrossEntropyLoss")
    print("二分类 → BCELoss (配合 Sigmoid)")
    print("回归 → MSELoss 或 L1Loss")
    print("目标检测 → 组合损失 (BCE + L1/SmoothL1)")

if __name__ == "__main__":
    main()
