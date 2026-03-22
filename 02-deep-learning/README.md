# 02-Deep-Learning - 深度学习基础

> 学习深度学习和 CNN 的基础知识

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 理解神经网络基础
- ✅ 掌握 CNN 工作原理
- ✅ 理解损失函数和优化器
- ✅ 使用 PyTorch 构建简单网络

---

## 📚 1. 神经网络基础

### 1.1 什么是神经网络

神经网络是一种模拟人脑的计算模型，由多层神经元组成。

```
输入层 → 隐藏层 → 输出层
```

**基本概念：**

| 术语 | 说明 | 示例 |
|------|------|------|
| 神经元 | 基本计算单元 | 接收输入，产生输出 |
| 权重 | 连接强度 | 决定输入的重要性 |
| 偏置 | 激活阈值 | 调整神经元敏感度 |
| 激活函数 | 非线性变换 | ReLU, Sigmoid, Softmax |

### 1.2 常用激活函数

```python
# examples/01-activation-functions.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ReLU (最常用)
relu = nn.ReLU()
# f(x) = max(0, x)
# 优点：计算快，梯度不消失

# Sigmoid
sigmoid = nn.Sigmoid()
# f(x) = 1 / (1 + exp(-x))
# 输出范围：0-1，用于二分类

# Softmax (多分类)
softmax = nn.Softmax(dim=1)
# 输出概率分布

print("激活函数说明:")
print("ReLU: 最常用，计算快")
print("Sigmoid: 二分类输出层")
print("Softmax: 多分类输出层")
```

---

## 🏗️ 2. 卷积神经网络 (CNN)

### 2.1 CNN 为什么适合图像处理

**全连接网络的问题：**
- 参数太多
- 忽略空间结构

**CNN 的优势：**
- 局部连接（关注局部特征）
- 权值共享（减少参数）
- 平移不变性

### 2.2 CNN 核心组件

```
输入图像
    ↓
[卷积层] → 提取特征
    ↓
[池化层] → 降维
    ↓
[全连接层] → 分类
    ↓
输出
```

### 2.3 卷积层 (Convolutional Layer)

**卷积操作：** 用卷积核在图像上滑动，提取特征

```python
# examples/02-convolution.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2D 卷积
# input: (batch_size, in_channels, height, width)
conv = nn.Conv2d(
    in_channels=3,    # 输入通道数（RGB）
    out_channels=32,  # 输出通道数（特征图数量）
    kernel_size=3,    # 卷积核大小（3x3）
    stride=1,         # 步长
    padding=1         # 填充（保持尺寸）
)

# 输入：(1, 3, 224, 224)
x = torch.randn(1, 3, 224, 224)

# 输出：(1, 32, 224, 224)
out = conv(x)

print(f"输入形状：{x.shape}")
print(f"输出形状：{out.shape}")

# 尺寸计算公式
# output_size = (input_size - kernel_size + 2*padding) / stride + 1
```

### 2.4 池化层 (Pooling Layer)

**作用：** 降维，减少计算量，防止过拟合

```python
# examples/03-pooling.py
import torch
import torch.nn as nn

# 最大池化（最常用）
max_pool = nn.MaxPool2d(
    kernel_size=2,  # 池化窗口大小
    stride=2        # 步长
)

# 平均池化
avg_pool = nn.AvgPool2d(
    kernel_size=2,
    stride=2
)

# 输入：(1, 32, 100, 100)
x = torch.randn(1, 32, 100, 100)

# 输出：(1, 32, 50, 50)
out_max = max_pool(x)
out_avg = avg_pool(x)

print(f"输入形状：{x.shape}")
print(f"最大池化输出：{out_max.shape}")
print(f"平均池化输出：{out_avg.shape}")
```

---

## 📦 3. 完整的 CNN 模型

### 3.1 使用 PyTorch 构建 CNN

```python
# examples/04-simple-cnn.py
"""
构建一个简单的 CNN 模型
用于图像分类
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

        # 卷积层 2
        # 输入：(32, 32, 32) → 输出：(64, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        # 输入：64 * 16 * 16 → 输出：128
        self.fc1 = nn.Linear(64 * 16 * 16, 128)

        # 输出层
        # 输入：128 → 输出：num_classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """前向传播"""
        # 卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))  # (32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (64, 8, 8)

        # 展平
        x = x.view(-1, 64 * 8 * 8)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# 创建模型
model = SimpleCNN(num_classes=10)
print(f"模型结构:\n{model}")

# 测试前向传播
x = torch.randn(1, 3, 32, 32)
out = model(x)
print(f"\n输入形状：{x.shape}")
print(f"输出形状：{out.shape}")
```

---

## 📊 4. 损失函数和优化器

### 4.1 常用损失函数

```python
# examples/05-loss-functions.py
import torch
import torch.nn as nn

# 1. 交叉熵损失（分类最常用）
# 用于多分类问题
ce_loss = nn.CrossEntropyLoss()
predictions = torch.randn(3, 5)  # 3 个样本，5 个类别
targets = torch.tensor([1, 0, 4])  # 真实标签
loss = ce_loss(predictions, targets)
print(f"交叉熵损失：{loss.item():.4f}")

# 2. 均方误差（回归问题）
mse_loss = nn.MSELoss()
predictions = torch.randn(3, 1)
targets = torch.randn(3, 1)
loss = mse_loss(predictions, targets)
print(f"均方误差：{loss.item():.4f}")

# 3. 二元交叉熵（二分类）
bce_loss = nn.BCELoss()
predictions = torch.sigmoid(torch.randn(3, 1))
targets = torch.randn(3, 1)
loss = bce_loss(predictions, targets)
print(f"二元交叉熵：{loss.item():.4f}")
```

### 4.2 常用优化器

```python
# examples/06-optimizers.py
import torch
import torch.nn as nn
import torch.optim as optim

# 创建模型
model = nn.Linear(10, 2)

# 1. SGD (随机梯度下降)
# 最经典，可配合动量
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 2. Adam (最常用)
# 自适应学习率，收敛快
adam = optim.Adam(model.parameters(), lr=0.001)

# 3. AdamW
# Adam 的改进版，权重衰减更好
adamw = optim.AdamW(model.parameters(), lr=0.001)

print("优化器选择建议:")
print("- Adam: 默认选择，适合大多数情况")
print("- SGD+ 动量：需要精细调参时用")
print("- AdamW：训练更稳定")
```

### 4.3 完整的训练循环

```python
# examples/07-training-loop.py
"""
完整的训练循环示例
"""
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据（示例用随机数据）
X_train = torch.randn(100, 10)  # 100 个样本，10 个特征
y_train = torch.randint(0, 2, (100,))  # 二分类标签

# 2. 创建模型
model = nn.Linear(10, 2)

# 3. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 4. 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 训练循环
num_epochs = 10

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 打印进度
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成!")
```

---

## 🚀 5. 迁移学习

### 5.1 使用预训练模型

```python
# examples/08-transfer-learning.py
"""
使用预训练模型进行迁移学习
"""
import torch
import torchvision.models as models

# 1. 加载预训练模型（ResNet18）
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 2. 冻结参数（可选）
for param in resnet.parameters():
    param.requires_grad = False

# 3. 修改最后的全连接层
# 假设我们有 5 个类别
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 5)

# 4. 现在可以训练新的分类层
print(f"模型已修改，输出类别数：5")
print(f"需要训练的参数：{sum(p.numel() for p in resnet.fc.parameters())}")
```

---

## 📝 实战练习

### 练习 1：理解 CNN 结构（30 分钟）

```python
# exercises/01-cnn-structure.py
"""
分析给定 CNN 模型的输出尺寸

任务：计算每个层的输出尺寸
"""

# 模型结构:
# 输入：(3, 224, 224)
# Conv2d(3, 64, 3, padding=1) → ?
# MaxPool2d(2, 2) → ?
# Conv2d(64, 128, 3, padding=1) → ?
# MaxPool2d(2, 2) → ?
# Conv2d(128, 256, 3, padding=1) → ?
# MaxPool2d(2, 2) → ?
# 全连接层输入维度 = ?

# TODO: 计算并填写每个层的输出尺寸
```

### 练习 2：构建简单分类器（60 分钟）

```python
# exercises/02-build-classifier.py
"""
构建一个 CNN 分类器

任务:
1. 定义 CNN 模型（2 个卷积层 + 全连接层）
2. 准备数据集（使用 torchvision 的 CIFAR-10）
3. 训练模型
4. 评估准确率
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# TODO: 完成练习
```

### 练习 3：使用预训练模型（30 分钟）

```python
# exercises/03-use-pretrained.py
"""
使用预训练模型进行图像分类

任务:
1. 加载预训练的 ResNet
2. 准备一张图片
3. 运行推理
4. 输出预测结果
"""
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# TODO: 完成练习
```

---

## ✅ 学习检查清单

完成本章后，确保你能够：

- [ ] 解释神经网络的基本原理
- [ ] 说明 CNN 为什么适合图像处理
- [ ] 写出卷积层和池化层的作用
- [ ] 使用 PyTorch 构建简单 CNN
- [ ] 选择合适的损失函数
- [ ] 使用优化器训练模型
- [ ] 理解迁移学习的概念

---

## 🔗 相关资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [深度学习入门](https://www.deeplearningbook.org/)
- [CNN 可视化](https://poloclub.github.io/cnn-explainer/)
- [PyTorch 中文文档](https://pytorch-cn.readthedocs.io/)

---

**下一步：[03-Object-Detection - 目标检测基础](../03-object-detection/README.md)** 🚀
