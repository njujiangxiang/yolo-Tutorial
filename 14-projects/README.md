# 14-Projects - 实战项目

> 完整项目实战：从数据收集到模型部署

---

## 🎯 项目列表

| 项目 | 难度 | 学时 | 内容 |
|------|------|------|------|
| 项目 1：PCB 缺陷检测 | ⭐⭐⭐ | 4 小时 | 完整流程实战 |
| 项目 2：表面划痕检测 | ⭐⭐ | 3 小时 | 小目标检测 |
| 项目 3：产品装配检测 | ⭐⭐ | 2 小时 | 多类别检测 |

---

## 项目 1：PCB 缺陷检测系统

**难度：** ⭐⭐⭐⭐
**预计时间：** 4-6 小时
**目标：** 构建完整的 PCB 缺陷检测系统

### 项目背景

PCB（印制电路板）生产过程中需要检测各种缺陷：
- 孔洞（hole）
- 划痕（scratch）
- 短路（short）
- 断路（open）
- 多余铜皮（spurious）

### 步骤 1：数据准备

#### 1.1 下载数据集

```bash
# 使用公开数据集
# PCB Defect Dataset: https://github.com/CharlesAverill/PCB-defect-detection

git clone https://github.com/CharlesAverill/PCB-defect-detection.git datasets/pcb-raw/
```

#### 1.2 数据整理

```python
# projects/pcb-defect/organize_data.py
"""
整理 PCB 缺陷数据集
"""
import os
import shutil
from pathlib import Path
import random

def organize_pcb_dataset(source_dir, output_dir):
    """
    将原始 PCB 数据整理为 YOLO 格式

    目录结构:
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
    """
    # 创建目录
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # PCB 数据集通常包含图片和对应的标注文件
    # 需要根据具体数据集格式进行解析

    # 这里提供通用模板
    image_files = list(Path(source_dir).glob("*.jpg"))
    random.shuffle(image_files)

    # 80% 训练集，20% 验证集
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # 复制文件
    for split, files in [('train', train_files), ('val', val_files)]:
        for img_path in files:
            shutil.copy(str(img_path), f"{output_dir}/images/{split}/")

            # 如果有对应的标注文件
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy(str(label_path), f"{output_dir}/labels/{split}/")

    print(f"整理完成：{output_dir}")
    print(f"  训练集：{len(train_files)} 张")
    print(f"  验证集：{len(val_files)} 张")

if __name__ == "__main__":
    organize_pcb_dataset(
        source_dir='datasets/pcb-raw/images/',
        output_dir='datasets/pcb-yolo/'
    )
```

#### 1.3 创建配置文件

```yaml
# projects/pcb-defect/data.yaml
path: /absolute/path/to/datasets/pcb-yolo
train: images/train
val: images/val

nc: 5
names:
  - hole        # 孔洞
  - scratch     # 划痕
  - short       # 短路
  - open        # 断路
  - spurious    # 多余铜皮
```

### 步骤 2：模型训练

```python
# projects/pcb-defect/train.py
"""
训练 PCB 缺陷检测模型
"""
from ultralytics import YOLO

def train_pcb_defect_model():
    # 1. 加载预训练模型
    model = YOLO('yolov8s.pt')  # 使用 small 模型

    # 2. 训练
    results = model.train(
        data='data.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        optimizer='SGD',
        lr0=0.01,
        patience=50,
        amp=True,
        project='runs/detect',
        name='pcb-defect',
        plots=True
    )

    # 3. 保存最佳模型
    best_model_path = 'runs/detect/pcb-defect/weights/best.pt'
    print(f"训练完成，模型保存在：{best_model_path}")

    return results

if __name__ == "__main__":
    train_pcb_defect_model()
```

```bash
# 运行训练
cd projects/pcb-defect
python train.py

# 或使用命令行
yolo detect train data=data.yaml model=yolov8s.pt epochs=150
```

### 步骤 3：模型评估

```python
# projects/pcb-defect/evaluate.py
"""
评估 PCB 缺陷检测模型
"""
from ultralytics import YOLO
import json

def evaluate_model():
    # 加载最佳模型
    model = YOLO('runs/detect/pcb-defect/weights/best.pt')

    # 运行验证
    metrics = model.val(data='data.yaml', plots=True)

    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print("=" * 50)

    # 保存结果
    results = {
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map)
    }

    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存：evaluation_results.json")

if __name__ == "__main__":
    evaluate_model()
```

### 步骤 4：模型推理

```python
# projects/pcb-defect/detect.py
"""
PCB 缺陷检测推理
"""
from ultralytics import YOLO
import cv2
from pathlib import Path

def detect_pcb_defects(image_path, model_path):
    """
    检测 PCB 图像中的缺陷

    参数:
        image_path: 输入图片路径
        model_path: 模型路径

    返回:
        检测结果
    """
    # 加载模型
    model = YOLO(model_path)

    # 推理
    results = model.predict(image_path, conf=0.25)

    # 解析结果
    result = results[0]
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            'class_id': cls,
            'class_name': result.names[cls],
            'confidence': conf,
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })

    return detections

def batch_detect(input_dir, output_dir, model_path):
    """
    批量检测

    参数:
        input_dir: 输入图片目录
        output_dir: 输出目录
        model_path: 模型路径
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_files = list(Path(input_dir).glob("*.jpg"))

    for img_path in image_files:
        # 检测
        detections = detect_pcb_defects(str(img_path), model_path)

        # 打印结果
        print(f"\n{img_path.name}:")
        for det in detections:
            print(f"  {det['class_name']} ({det['confidence']:.2f})")

    print(f"\n批量检测完成")

if __name__ == "__main__":
    # 单张图片检测
    detections = detect_pcb_defects(
        image_path='test_pcb.jpg',
        model_path='runs/detect/pcb-defect/weights/best.pt'
    )

    print("检测结果:")
    for det in detections:
        print(f"  {det['class_name']}: {det['confidence']:.2f}")

    # 批量检测
    # batch_detect('datasets/val/images/', 'results/', model_path)
```

### 步骤 5：结果可视化

```python
# projects/pcb-defect/visualize.py
"""
可视化检测结果
"""
from ultralytics import YOLO
import cv2
import numpy as np

def visualize_detection(image_path, model_path, output_path):
    """
    可视化检测并保存结果图片
    """
    model = YOLO(model_path)

    # 推理
    results = model.predict(image_path, conf=0.25)
    result = results[0]

    # 绘制结果
    result_image = result.plot()

    # 保存
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存：{output_path}")

def create_comparison(image_path, model_path, output_path):
    """
    创建对比图（原图 vs 检测结果）
    """
    # 读取原图
    original = cv2.imread(image_path)

    # 推理
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.25)
    result_image = results[0].plot()

    # 创建对比图
    # 确保尺寸相同
    h, w = original.shape[:2]
    result_image = cv2.resize(result_image, (w, h))

    # 并排显示
    comparison = np.hstack([original, result_image])

    # 添加标签
    cv2.putText(comparison, 'Original', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, 'Detection', (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(output_path, comparison)
    print(f"对比图已保存：{output_path}")

if __name__ == "__main__":
    visualize_detection(
        'test_pcb.jpg',
        'runs/detect/pcb-defect/weights/best.pt',
        'detection_result.jpg'
    )

    create_comparison(
        'test_pcb.jpg',
        'runs/detect/pcb-defect/weights/best.pt',
        'comparison.jpg'
    )
```

### 步骤 6：导出部署

```python
# projects/pcb-defect/export.py
"""
导出模型用于部署
"""
from ultralytics import YOLO

def export_models():
    """
    导出为多种格式
    """
    model = YOLO('runs/detect/pcb-defect/weights/best.pt')

    # 导出为 ONNX（推荐）
    onnx_path = model.export(format='onnx', simplify=True)
    print(f"ONNX 模型：{onnx_path}")

    # 导出为 TensorRT（NVIDIA GPU）
    # trt_path = model.export(format='engine', device=0)
    # print(f"TensorRT 模型：{trt_path}")

    # 导出为 OpenVINO（Intel CPU）
    # openvino_path = model.export(format='openvino')
    # print(f"OpenVINO 模型：{openvino_path}")

if __name__ == "__main__":
    export_models()
```

---

## 项目 2：表面划痕检测

**难度：** ⭐⭐⭐
**预计时间：** 3-4 小时

### 项目概述

检测金属/玻璃表面的划痕缺陷，属于小目标检测问题。

### 关键挑战

1. **划痕很细小** - 可能只有几个像素宽
2. **对比度低** - 划痕与背景颜色接近
3. **方向多变** - 划痕可能朝任何方向

### 解决方案

```python
# projects/surface-scratch/train.py
"""
表面划痕检测训练配置
"""
from ultralytics import YOLO

def train_scratch_detection():
    model = YOLO('yolov8m.pt')  # 使用 medium 模型检测小目标

    results = model.train(
        data='scratch_data.yaml',
        epochs=200,
        imgsz=800,  # 增大输入尺寸
        batch=8,    # 减小批次
        device=0,
        mosaic=0.5,  # 降低 mosaic 比例
        close_mosaic=50,  # 最后 50 轮关闭 mosaic
        lr0=0.001,
        patience=100,
    )

    return results
```

---

## 项目 3：产品装配检测

**难度：** ⭐⭐
**预计时间：** 2-3 小时

### 项目概述

检测产品装配是否完整，例如：
- 螺丝是否安装
- 标签是否粘贴
- 零件是否缺失

### 训练配置

```yaml
# projects/assembly-check/data.yaml
nc: 4
names:
  - screw       # 螺丝
  - label       # 标签
  - cap         # 盖子
  - missing_part  # 缺失零件
```

---

## 📝 实战练习

### 完整项目练习（4-6 小时）

完成 PCB 缺陷检测全流程：

```bash
# 1. 准备数据（30 分钟）
cd projects/pcb-defect
python organize_data.py

# 2. 训练模型（60-90 分钟）
python train.py

# 3. 评估模型（30 分钟）
python evaluate.py

# 4. 推理测试（30 分钟）
python detect.py

# 5. 可视化结果（30 分钟）
python visualize.py

# 6. 导出模型（15 分钟）
python export.py
```

### 项目报告模板

完成项目后，创建项目报告 `project_report.md`：

```markdown
# PCB 缺陷检测项目报告

## 项目概述
- 检测目标：PCB 板上的 5 种缺陷
- 数据集：X 张图片，Y 张训练，Z 张验证
- 模型：YOLOv8s

## 训练结果
- 训练轮数：150
- 最终损失：X.XXX
- 训练时间：XX 分钟

## 评估结果
| 指标 | 数值 |
|------|------|
| Precision | 0.XX |
| Recall | 0.XX |
| mAP50 | 0.XX |
| mAP50-95 | 0.XX |

## 各类别表现
| 类别 | Precision | Recall | mAP50 |
|------|-----------|--------|-------|
| hole | | | |
| scratch | | | |
| short | | | |
| open | | | |
| spurious | | | |

## 坏案例分析
- 主要漏检类型：XXX
- 主要误检类型：XXX
- 改进方向：XXX

## 部署方案
- 导出格式：ONNX
- 推理速度：XX ms/张
- 部署平台：XXX

## 总结
成功经验：
1. ...
2. ...

待改进：
1. ...
2. ...
```

---

## ✅ 项目完成检查清单

完成项目后，确保：

- [ ] 数据准备完成
- [ ] 模型训练完成，mAP50 > 0.7
- [ ] 评估报告生成
- [ ] 推理脚本可运行
- [ ] 可视化结果正常
- [ ] 模型成功导出
- [ ] 项目报告完成

---

**恭喜你完成 YOLO 培训！现在你已经具备了独立开展缺陷检测项目的能力。** 🎉

---

## 🔗 相关资源

- [Ultralytics 项目示例](https://github.com/ultralytics/ultralytics/tree/main/examples)
- [PCB 缺陷检测论文](https://ieeexplore.ieee.org/document/8946770)
- [工业缺陷检测综述](https://www.sciencedirect.com/science/article/pii/S0736584520300033)
