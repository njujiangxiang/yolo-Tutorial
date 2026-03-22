# 11-Model Evaluation - 模型评估与诊断

> 学习如何评估 YOLO 模型性能和诊断问题

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 理解评估指标（precision, recall, mAP）
- ✅ 解读混淆矩阵
- ✅ 分析训练曲线
- ✅ 找出并分析坏案例
- ✅ 诊断常见训练问题
- ✅ 生成评估报告

---

## 📊 评估指标详解

### 基础概念

在目标检测中，每个预测可以分为：

| 预测结果 | 实际有目标 | 实际无目标 |
|----------|-----------|-----------|
| 预测有目标 | **True Positive (TP)** | **False Positive (FP)** |
| 预测无目标 | **False Negative (FN)** | **True Negative (TN)** |

**缺陷检测中的例子：**
- **TP**: 真的有划痕，模型也检测到了 ✓
- **FP**: 没有划痕，模型误报了 ✗
- **FN**: 有划痕，模型没检测到 ✗（最危险！）
- **TN**: 没有划痕，模型也没报错 ✓

---

### Precision（查准率）

**定义：** 预测为正样本中，实际为正的比例

$$Precision = \frac{TP}{TP + FP}$$

**含义：** 模型报告的缺陷中，有多少是真的缺陷

**例子：**
```
模型检测到 100 个缺陷
其中 85 个是真的缺陷，15 个是误报

Precision = 85 / (85 + 15) = 0.85 = 85%
```

**何时重要：** 误报成本高时（如误报会导致停机检查）

---

### Recall（查全率）

**定义：** 实际正样本中，被正确预测的比例

$$Recall = \frac{TP}{TP + FN}$$

**含义：** 所有真实缺陷中，模型找到了多少

**例子：**
```
图片中有 50 个真实缺陷
模型找到了 40 个，漏了 10 个

Recall = 40 / (40 + 10) = 0.80 = 80%
```

**何时重要：** 漏检成本高时（如安全检测、缺陷检测）

---

### Precision-Recall 权衡

**不可能同时最大化两者！**

| 场景 | Precision | Recall | 适用 |
|------|-----------|--------|------|
| 高置信度阈值 | 高 | 低 | 减少误报 |
| 低置信度阈值 | 低 | 高 | 减少漏检 |

**缺陷检测推荐：** 优先保证 Recall

```python
# 降低置信度阈值，减少漏检
results = model.predict(conf=0.25)  # 默认 0.25

# 如果漏检多，降低阈值
results = model.predict(conf=0.15)

# 如果误报多，提高阈值
results = model.predict(conf=0.50)
```

---

### IoU（Intersection over Union）

**定义：** 预测框和真实框的交并比

$$IoU = \frac{Area_{overlap}}{Area_{union}}$$

**示例：**
```
预测框和真实框完全重合：IoU = 1.0
预测框和真实框一半重合：IoU = 0.5
预测框和真实框不重合：IoU = 0
```

**阈值：**
- IoU ≥ 0.5：通常认为是正确检测
- IoU ≥ 0.75：严格正确检测
- IoU ≥ 0.95：非常精确

---

### mAP（mean Average Precision）

**mAP50：** IoU 阈值为 0.5 时的 mAP

**mAP50-95：** IoU 从 0.5 到 0.95（步长 0.05）的平均 mAP

**解释：**
```
mAP50 = 0.85
意思：在 IoU 阈值 0.5 时，平均精度为 85%

mAP50-95 = 0.55
意思：在严格标准下（IoU 0.5-0.95），平均精度为 55%
```

**评估标准：**

| mAP50 | mAP50-95 | 评价 |
|-------|----------|------|
| >0.90 | >0.70 | 优秀 |
| >0.80 | >0.50 | 良好 |
| >0.70 | >0.40 | 可用 |
| <0.70 | <0.40 | 需要改进 |

---

## 📈 混淆矩阵

### 混淆矩阵示例

```
混淆矩阵（归一化）

                   预测
              正常   划痕  孔洞  短路
        正常  0.95  0.03  0.01  0.01
    实  划痕  0.02  0.90  0.05  0.03
    际  孔洞  0.01  0.08  0.88  0.03
        短路  0.02  0.05  0.03  0.90
```

**解读：**
- 对角线：正确分类的比例
- 非对角线：混淆的情况
- 例如：0.08 表示 8% 的孔洞被误判为划痕

### 生成混淆矩阵

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/defect-train/weights/best.pt')

# 生成混淆矩阵
model.val(data='datasets/labeled/data.yaml', plots=True)

# 混淆矩阵图片保存在：
# runs/detect/defect-train/confusion_matrix.png
```

### 混淆矩阵分析代码

```python
# examples/analyze-confusion-matrix.py
import numpy as np
import matplotlib.pyplot as plt

def analyze_confusion_matrix(matrix, class_names):
    """
    分析混淆矩阵

    参数:
        matrix: 混淆矩阵 numpy 数组
        class_names: 类别名称列表
    """
    # 计算各类别的召回率
    recalls = np.diag(matrix) / matrix.sum(axis=1)

    # 计算各类别的精确率
    precisions = np.diag(matrix) / matrix.sum(axis=0)

    print("类别分析:")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"{name:10s}: Recall={recalls[i]:.2%}, Precision={precisions[i]:.2%}")

    # 找出最容易混淆的类别对
    np.fill_diagonal(matrix, 0)  # 忽略对角线
    max_confusion = np.unravel_index(np.argmax(matrix), matrix.shape)
    print(f"\n最容易混淆：{class_names[max_confusion[0]]} → {class_names[max_confusion[1]]}")
    print(f"混淆比例：{matrix[max_confusion]:.2%}")

if __name__ == "__main__":
    # 示例混淆矩阵
    matrix = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.02, 0.90, 0.05, 0.03],
        [0.01, 0.08, 0.88, 0.03],
        [0.02, 0.05, 0.03, 0.90]
    ])

    classes = ['正常', '划痕', '孔洞', '短路']
    analyze_confusion_matrix(matrix, classes)
```

---

## 📉 训练曲线分析

### 关键曲线

训练完成后，查看以下曲线：

1. **Loss 曲线** (`results.png`)
   - `box_loss`: 边界框损失
   - `cls_loss`: 分类损失
   - `dfl_loss`: 分布焦点损失

2. **mAP 曲线** (`results.png`)
   - `mAP50`: IoU=0.5 时的 mAP
   - `mAP50-95`: 严格 mAP

3. **Precision-Recall 曲线** (`PR_curve.png`)

### 曲线解读

**正常收敛：**
```
Loss 曲线：
  ↑
  │┌──────────
  ││
  │ ──────────  ← 平稳收敛
  └────────────→ epoch

mAP 曲线：
  ↑    ┌────────
  │   /
  │  /
  │ /
  └────────────→ epoch
```

**过拟合：**
```
训练 loss 持续下降
验证 loss 在某个点开始上升

  ↑   训练 loss
  │  /
  │ /
  │/──────────
  │      ╲
  │       ╲ 验证 loss
  └────────────→ epoch
```

**欠拟合：**
```
Loss 还在下降但停止了
mAP 还在上升但停止了

 应该继续训练！
```

### 可视化训练结果

```python
# examples/plot-training-results.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_results(results_csv_path):
    """
    绘制训练结果曲线
    """
    # 读取结果
    df = pd.read_csv(results_csv_path)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Loss 曲线
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='box_loss')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='cls_loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training Loss')

    # 2. mAP 曲线
    axes[0, 1].plot(df['epoch'], df['metrics/precision'], label='precision')
    axes[0, 1].plot(df['epoch'], df['metrics/recall'], label='recall')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].set_title('Metrics')

    # 3. mAP50
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50'], label='mAP50', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP50')
    axes[1, 0].legend()
    axes[1, 0].set_title('mAP50')

    # 4. mAP50-95
    axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95'], label='mAP50-95', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP50-95')
    axes[1, 1].legend()
    axes[1, 1].set_title('mAP50-95')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_results('runs/detect/defect-train/results.csv')
```

---

## 🔍 坏案例分析

### 什么是坏案例？

**坏案例类型：**

| 类型 | 说明 | 例子 |
|------|------|------|
| 漏检 (FN) | 有缺陷但没检测到 | 小缺陷漏掉 |
| 误检 (FP) | 没缺陷但误报 | 灰尘误报为缺陷 |
| 定位不准 | 检测到但框不准 | IoU < 0.5 |
| 分类错误 | 缺陷类型判错 | 划痕判为孔洞 |

### 找出坏案例

```python
# examples/find-bad-cases.py
from ultralytics import YOLO
import cv2
import os

def find_false_negatives(model_path, data_yaml, output_dir):
    """
    找出漏检的案例
    """
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有验证集图片
    # ... 实现推理和对比
    # 找出有标注但没检测到的案例

    print(f"漏检案例保存到：{output_dir}")

def find_false_positives(model_path, data_yaml, output_dir):
    """
    找出误检的案例
    """
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # ... 实现检测但没标注的案例

    print(f"误检案例保存到：{output_dir}")

if __name__ == "__main__":
    find_false_negatives(
        'runs/detect/defect-train/weights/best.pt',
        'datasets/labeled/data.yaml',
        'bad_cases/false_negatives/'
    )

    find_false_positives(
        'runs/detect/defect-train/weights/best.pt',
        'datasets/labeled/data.yaml',
        'bad_cases/false_positives/'
    )
```

### 坏案例分析模板

```markdown
# 坏案例分析报告

## 案例 1：漏检

**图片：** `val_0012.jpg`

**问题：** 小划痕未检测到

**原因分析：**
- 划痕尺寸太小（<10 像素）
- 对比度低，不明显

**改进方案：**
1. 增加小样本的权重
2. 降低输入尺寸阈值
3. 增加针对小目标的增强

---

## 案例 2：误检

**图片：** `val_0025.jpg`

**问题：** 灰尘误报为划痕

**原因分析：**
- 灰尘呈线状
- 训练数据中类似灰尘的样本少

**改进方案：**
1. 在训练数据中加入灰尘样本（标注为背景）
2. 提高置信度阈值
```

---

## 🩺 常见问题诊断

### 问题 1：某类缺陷检测效果差

**现象：** 混淆矩阵显示某类 Recall 特别低

**原因：**
1. 该类样本太少
2. 该类特征不明显
3. 标注不一致

**解决方法：**

```python
# 1. 针对性增加数据
# 收集更多该类缺陷的图片

# 2. 过采样
# 对含该类缺陷的图片多增强几倍

# 3. 检查标注
# 确保该类标注准确一致
```

---

### 问题 2：小缺陷检测不到

**现象：** 小目标（<20x20 像素）Recall 低

**解决方法：**

```python
# 1. 增大输入尺寸
model.train(imgsz=800)

# 2. 关闭 Mosaic（会损失小目标）
model.train(mosaic=0.5)

# 3. 使用专门的小目标检测配置
model.train(
    imgsz=1280,
    batch=4,
    close_mosaic=50,  # 最后 50 轮关闭 mosaic
)
```

---

### 问题 3：误报太多

**现象：** Precision 低，FP 多

**解决方法：**

```python
# 1. 提高置信度阈值
results = model.predict(conf=0.5)  # 从 0.25 提高到 0.5

# 2. 增加负样本训练
# 收集不含缺陷的图片，标注为空

# 3. 检查训练数据是否有误标
```

---

## 📋 生成评估报告

### 完整评估脚本

```python
# examples/generate-eval-report.py
from ultralytics import YOLO
import yaml
import json
from datetime import datetime

def generate_evaluation_report(model_path, data_yaml, output_path):
    """
    生成完整的评估报告
    """
    # 加载模型和数据
    model = YOLO(model_path)
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # 运行验证
    metrics = model.val(data=data_yaml, plots=True, save_json=True)

    # 生成报告
    report = {
        'model': model_path,
        'dataset': data_yaml,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
        },
        'per_class': []
    }

    # 各类别指标
    for i, name in enumerate(data['names']):
        report['per_class'].append({
            'class_id': i,
            'class_name': name,
            'precision': float(metrics.box.mp[i]),
            'recall': float(metrics.box.mr[i]),
            'mAP50': float(metrics.box.map50[i]),
        })

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"评估报告已保存：{output_path}")

    # 打印摘要
    print("\n" + "=" * 50)
    print("评估摘要")
    print("=" * 50)
    print(f"Precision:  {report['metrics']['precision']:.2%}")
    print(f"Recall:     {report['metrics']['recall']:.2%}")
    print(f"mAP50:      {report['metrics']['mAP50']:.2%}")
    print(f"mAP50-95:   {report['metrics']['mAP50-95']:.2%}")
    print("=" * 50)

    return report

if __name__ == "__main__":
    generate_evaluation_report(
        model_path='runs/detect/defect-train/weights/best.pt',
        data_yaml='datasets/labeled/data.yaml',
        output_path='evaluation_report.json'
    )
```

---

## 📝 实战练习

### 练习 1：生成评估报告（30 分钟）

```bash
# 1. 运行评估
python examples/generate-eval-report.py

# 2. 查看各项指标
# 3. 记录 mAP50 和 mAP50-95
```

### 练习 2：分析混淆矩阵（30 分钟）

```bash
# 1. 查看混淆矩阵图片
# runs/detect/defect-train/confusion_matrix.png

# 2. 运行分析脚本
python examples/analyze-confusion-matrix.py

# 3. 找出最容易混淆的类别
```

### 练习 3：分析训练曲线（30 分钟）

```bash
# 1. 查看 results.png
# 2. 运行绘图脚本
python examples/plot-training-results.py

# 3. 判断是否收敛，是否过拟合
```

### 练习 4：坏案例分析（60 分钟）

```bash
# 1. 找出漏检案例
python examples/find-bad-cases.py

# 2. 人工检查 10 个坏案例
# 3. 分析原因并记录
```

---

## ✅ 评估检查清单

评估完成后，确保：

- [ ] mAP50 > 0.7（可用）
- [ ] mAP50-95 > 0.4（可用）
- [ ] 各类别 Recall 均衡
- [ ] 没有明显的类别混淆
- [ ] 训练曲线正常收敛
- [ ] 分析了坏案例并记录原因

---

## 🔗 相关资源

- [目标检测评估指标详解](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [Ultralytics 验证文档](https://docs.ultralytics.com/modes/val/)
- [混淆矩阵教程](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)

---

**下一步：[12-Model-Deployment - 模型部署](../12-model-deployment/README.md)** 🚀
