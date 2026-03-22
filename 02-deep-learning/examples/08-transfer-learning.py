# examples/08-transfer-learning.py
"""
迁移学习

使用预训练模型进行迁移学习
"""
import torch
import torchvision.models as models
import torch.nn as nn

def main():
    print("=" * 50)
    print("迁移学习演示")
    print("=" * 50)

    # 方法 1: 使用预训练模型做推理
    print("\n方法 1: 直接使用预训练模型")
    print("-" * 50)

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.eval()

    print(f"模型：ResNet18 (ImageNet 预训练)")
    print(f"输入形状：(batch, 3, 224, 224)")

    # 测试推理
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = resnet(x)
    print(f"输出形状：{output.shape}")
    print(f"输出含义：1000 个 ImageNet 类别的分数")

    # 方法 2: 微调最后层
    print("\n方法 2: 微调最后层")
    print("-" * 50)

    # 加载预训练模型
    resnet_finetune = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 冻结所有参数
    for param in resnet_finetune.parameters():
        param.requires_grad = False

    # 修改最后的全连接层
    # 假设我们有 5 个类别（如缺陷检测）
    num_features = resnet_finetune.fc.in_features
    resnet_finetune.fc = nn.Linear(num_features, 5)

    # 现在只有新的全连接层可以训练
    trainable_params = sum(p.numel() for p in resnet_finetune.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in resnet_finetune.parameters())

    print(f"修改后：最后层输出 5 个类别")
    print(f"总参数：{total_params:,}")
    print(f"可训练参数：{trainable_params:,}")
    print(f"训练比例：{100 * trainable_params / total_params:.1f}%")

    # 方法 3: 部分微调
    print("\n方法 3: 部分微调")
    print("-" * 50)

    resnet_partial = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 只冻结前面几层
    for name, param in resnet_partial.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # 修改最后层
    resnet_partial.fc = nn.Linear(resnet_partial.fc.in_features, 5)

    trainable_params = sum(p.numel() for p in resnet_partial.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in resnet_partial.parameters())

    print(f"解冻 layer4 和 fc 层")
    print(f"可训练参数：{trainable_params:,} / {total_params:,}")
    print(f"训练比例：{100 * trainable_params / total_params:.1f}%")

    print("\n" + "=" * 50)
    print("迁移学习建议:")
    print("=" * 50)
    print("1. 数据少 → 冻结全部，只训练最后层")
    print("2. 数据中等 → 冻结前面，微调后面")
    print("3. 数据多 → 全部微调")
    print("4. 缺陷检测 → 用 ImageNet 预训练效果很好")

if __name__ == "__main__":
    main()
