# examples/07-training-loop.py
"""
完整的训练循环

学习 PyTorch 标准训练流程
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def main():
    print("=" * 50)
    print("完整训练循环演示")
    print("=" * 50)

    # 1. 准备数据
    print("\n步骤 1: 准备数据")
    X_train = torch.randn(1000, 10)  # 1000 个样本，10 个特征
    y_train = torch.randint(0, 2, (1000,))  # 二分类标签

    # 创建 DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"   训练集大小：{len(dataset)}")
    print(f"   批次大小：32")
    print(f"   批次数量：{len(dataloader)}")

    # 2. 创建模型
    print("\n步骤 2: 创建模型")
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    print(f"   模型结构：10 → 64 → 32 → 2")

    # 3. 定义损失函数
    print("\n步骤 3: 定义损失函数")
    criterion = nn.CrossEntropyLoss()
    print(f"   损失函数：CrossEntropyLoss")

    # 4. 定义优化器
    print("\n步骤 4: 定义优化器")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print(f"   优化器：Adam (lr=0.01)")

    # 5. 训练循环
    print("\n步骤 5: 训练")
    print("-" * 50)
    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        # 打印 epoch 结果
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"   Epoch [{epoch+1}/{num_epochs}]: "
              f"Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")

    print("-" * 50)
    print("\n训练完成!")

    # 6. 评估
    print("\n步骤 6: 评估")
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)
        _, predicted = torch.max(outputs, 1)
        accuracy = 100 * (predicted == y_train).sum().item() / len(y_train)
        print(f"   最终训练准确率：{accuracy:.1f}%")

if __name__ == "__main__":
    main()
