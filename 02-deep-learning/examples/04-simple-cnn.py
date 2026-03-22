# examples/04-simple-cnn.py
"""
构建简单的 CNN 模型

用于图像分类的完整 CNN 架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """简单的 CNN 分类模型"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # 卷积层 1
        # 输入：(3, 32, 32) → 输出：(32, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积层 2
        # 输入：(32, 32, 32) → 输出：(64, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        # 输入：64 * 16 * 16 → 输出：128
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)

        # 输出层
        # 输入：128 → 输出：num_classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """前向传播"""
        # 卷积块 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (32, 16, 16)

        # 卷积块 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (64, 8, 8)

        # 展平
        x = x.view(-1, 64 * 8 * 8)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def main():
    print("=" * 50)
    print("SimpleCNN 模型演示")
    print("=" * 50)

    # 创建模型
    model = SimpleCNN(num_classes=10)
    print(f"\n模型结构:\n{model}")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量：{total_params:,}")
    print(f"可训练参数：{trainable_params:,}")

    # 测试前向传播
    x = torch.randn(4, 3, 32, 32)  # 4 张图片
    out = model(x)
    print(f"\n输入形状：{x.shape}")
    print(f"输出形状：{out.shape}")
    print(f"输出含义：4 个样本，10 个类别的分数")

    # 转换为概率
    probs = F.softmax(out, dim=1)
    print(f"\n预测概率形状：{probs.shape}")
    print(f"第一个样本的预测：{probs[0]}")

if __name__ == "__main__":
    main()
