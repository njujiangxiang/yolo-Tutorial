# 09-Custom-Dataset - 数据标注实战

> 学习使用 LabelImg 标注数据，准备 YOLO 训练数据集

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 安装和配置 LabelImg
- ✅ 熟练使用标注快捷键
- ✅ 保证标注质量
- ✅ 将标注转换为 YOLO 格式
- ✅ 创建 data.yaml 配置文件

---

## 📋 标注前准备

### 目录结构

在开始标注前，确保目录结构如下：

```
project/
├── datasets/
│   ├── raw/              # 原始收集的图片
│   ├── cleaned/          # 清洗后的图片
│   └── labeled/          # 标注后的数据（输出）
│       ├── images/
│       └── labels/
└── classes.txt           # 类别文件
```

### 类别文件

创建 `classes.txt` 文件，定义要标注的类别：

```txt
# PCB 缺陷检测示例
hole
scratch
short
open
spurious
```

**注意事项：**
- 每行一个类别名称
- 不要有空格或特殊字符
- 类别顺序就是 ID 顺序（从 0 开始）

---

## 🛠️ 标注工具安装

### 方法 1：LabelImg（推荐）

```bash
# 安装
pip install labelImg

# 启动
labelImg
```

### 方法 2：从源码安装

```bash
# 克隆仓库
git clone https://github.com/heartexlabs/labelImg.git
cd labelImg

# 安装依赖
pip install -r requirements.txt

# 运行
python labelImg.py
```

### 方法 3：在线标注平台

如果不想安装本地软件，可以使用：

| 平台 | 网址 | 特点 |
|------|------|------|
| Roboflow | https://roboflow.com | 免费，支持导出 YOLO 格式 |
| CVAT | https://cvat.ai | Intel 出品，功能强大 |
| Label Studio | https://labelstud.io | 开源，支持多种任务 |

---

## 📖 LabelImg 使用教程

### 第一步：打开图片目录

1. 启动 LabelImg
2. 点击左侧 `Open Dir` 按钮
3. 选择 `datasets/cleaned/` 目录

### 第二步：设置标注保存目录

1. 点击 `Change Save Dir` 按钮
2. 选择 `datasets/labeled/labels/` 目录
3. 确保格式选择 `YOLO`

### 第三步：加载类别文件

1. 点击 `File` → `Save Format` → `YOLO`
2. 确保 `classes.txt` 在同一目录
3. 或者手动创建类别

### 第四步：开始标注

```
操作流程：
1. 按 W 键 → 出现画框工具
2. 框选目标 → 松开鼠标
3. 选择类别 → 点击 OK
4. 按 Ctrl+S → 保存
5. 按 D 键 → 下一张图片
```

---

## ⌨️ 标注快捷键

### 常用快捷键

| 快捷键 | 功能 | 说明 |
|--------|------|------|
| `W` | 画框 | 开始画矩形框 |
| `D` | 下一张 | 跳到下一张图片 |
| `A` | 上一张 | 跳到上一张图片 |
| `Ctrl+S` | 保存 | 保存当前标注 |
| `Ctrl+R` | 改颜色 | 改变框的颜色 |
| `Del` | 删除 | 删除选中的框 |
| `Ctrl+D` | 复制 | 复制选中的框 |
| `双击框` | 编辑 | 编辑框的位置和类别 |

### 效率技巧

1. **连续标注模式**：标注完一个框后，直接按 W 继续画下一个
2. **批量保存**：可以标注多张后统一保存（不推荐）
3. **快速切换类别**：在画框前选择类别

---

## ✅ 标注质量标准

### 好的标注

```
✅ 框紧贴目标边缘
✅ 包含完整目标（不遮挡的情况下）
✅ 小目标也标注（≥10x10 像素）
✅ 截断目标标注可见部分
✅ 类别选择正确
```

### 不好的标注

```
❌ 框太大，包含过多背景
❌ 框太小，漏掉目标边缘
❌ 漏标小目标
❌ 类别错误
❌ 同一个目标画多个框
```

### 标注示例对比

```
正确标注：          错误标注：
┌─────────────┐    ┌─────────────────┐
│   目标      │    │   目标          │
│   ████      │    │                 │
│   ████      │    │                 │
└─────────────┘    └─────────────────┘
  紧贴边缘              包含过多背景
```

---

## 🔄 标注格式转换

### YOLO 格式说明

**文件格式：** `.txt`
**每行一个目标：**
```
<class_id> <x_center> <y_center> <width> <height>
```

**值说明：**
- `class_id`: 类别 ID（从 0 开始）
- `x_center`: 框中心 X 坐标（归一化到 0-1）
- `y_center`: 框中心 Y 坐标（归一化到 0-1）
- `width`: 框宽度（归一化）
- `height`: 框高度（归一化）

### 归一化计算

```python
# 假设图片尺寸 640x480
# 原始框 [xmin, ymin, xmax, ymax] = [100, 100, 200, 200]

# 转换为 YOLO 格式
x_center = (100 + 200) / 2 / 640 = 0.234
y_center = (100 + 200) / 2 / 480 = 0.312
width = (200 - 100) / 640 = 0.156
height = (200 - 100) / 480 = 0.208

# YOLO 格式：0 0.234 0.312 0.156 0.208
```

### 转换脚本

```python
# examples/convert-to-yolo.py
import os
from pathlib import Path

def voc_to_yolo(xml_path, output_path, img_width, img_height):
    """
    将 VOC XML 转换为 YOLO 格式
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')

        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 转换
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    with open(output_path, 'w') as f:
        f.writelines(yolo_lines)
```

---

## 📊 标注质量检查

### 可视化检查

```python
# examples/visualize-labels.py
import cv2
import os
from pathlib import Path

def visualize_yolo_labels(image_path, label_path, output_path):
    """
    可视化 YOLO 标注
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return

    height, width = img.shape[:2]

    # 读取标注
    if not os.path.exists(label_path):
        return

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                w = float(parts[3]) * width
                h = float(parts[4]) * height

                # 转换为左上角坐标
                xmin = int(x_center - w / 2)
                ymin = int(y_center - h / 2)
                xmax = int(x_center + w / 2)
                ymax = int(y_center + h / 2)

                # 画框
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # 写类别
                cv2.putText(img, str(class_id), (xmin, ymin - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"可视化结果：{output_path}")

# 批量可视化
def batch_visualize(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_path in Path(image_dir).glob("*.jpg"):
        label_path = f"{label_dir}/{img_path.stem}.txt"
        visualize_yolo_labels(
            str(img_path),
            label_path,
            f"{output_dir}/{img_path.name}"
        )

if __name__ == "__main__":
    batch_visualize("datasets/labeled/images/",
                   "datasets/labeled/labels/",
                   "datasets/labeled/visualization/")
```

### 统计检查

```python
# examples/check-annotations.py
from pathlib import Path
from collections import defaultdict

def check_annotations(label_dir):
    """
    检查标注质量
    """
    stats = defaultdict(int)
    issues = []

    for label_file in Path(label_dir).glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            issues.append(f"空标注文件：{label_file}")
            continue

        for i, line in enumerate(lines):
            parts = line.strip().split()

            if len(parts) != 5:
                issues.append(f"格式错误：{label_file}:{i+1}")
                continue

            try:
                values = [float(p) for p in parts[1:]]
                if not all(0 <= v <= 1 for v in values):
                    issues.append(f"值超出范围：{label_file}:{i+1}")

                # 统计类别
                class_id = int(parts[0])
                stats[class_id] += 1

            except ValueError:
                issues.append(f"无效数值：{label_file}:{i+1}")

    # 输出报告
    print("=" * 50)
    print("标注检查报告")
    print("=" * 50)

    print(f"\n标注文件总数：{sum(stats.values())}")
    print("\n类别分布:")
    for class_id, count in sorted(stats.items()):
        print(f"  类别 {class_id}: {count} 个目标")

    if issues:
        print(f"\n⚠️ 发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前 10 个
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues) - 10} 个问题")
    else:
        print("\n✅ 没有发现问题")

    return len(issues) == 0

if __name__ == "__main__":
    check_annotations("datasets/labeled/labels/")
```

---

## 📝 创建 data.yaml

### 配置文件模板

```yaml
# datasets/labeled/data.yaml

# 数据集根目录路径（使用绝对路径）
path: /Users/xiaoyu/code/yolo-Tutorial/datasets/labeled

# 训练集和验证集路径（相对于 path）
train: images/train
val: images/val
test: images/test  # 可选

# 类别数
nc: 5

# 类别名称（按 ID 顺序）
names:
  - hole        # 0: 孔洞
  - scratch     # 1: 划痕
  - short       # 2: 短路
  - open        # 3: 断路
  - spurious    # 4: 多余物
```

### 自动生成脚本

```python
# examples/create-data-yaml.py
import yaml
import os

def create_data_yaml(dataset_dir, class_names, output_path):
    """
    创建 YOLO data.yaml 配置文件
    """
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"data.yaml 已创建：{output_path}")
    print("\n内容:")
    print(yaml.dump(config, allow_unicode=True))

if __name__ == "__main__":
    # PCB 缺陷检测
    classes = ['hole', 'scratch', 'short', 'open', 'spurious']

    create_data_yaml(
        dataset_dir="datasets/labeled/",
        class_names=classes,
        output_path="datasets/labeled/data.yaml"
    )
```

---

## 📝 实战练习

### 练习 1：LabelImg 安装与配置（15 分钟）
```bash
# 1. 安装 LabelImg
pip install labelImg

# 2. 启动
labelImg

# 3. 熟悉界面和快捷键
```

### 练习 2：标注 50 张图片（60 分钟）
```
1. 打开 datasets/cleaned/ 目录
2. 标注所有图片
3. 确保每张图片标注准确
4. 保存到 datasets/labeled/labels/
```

### 练习 3：标注质量检查（30 分钟）
```bash
# 1. 运行检查脚本
python examples/check-annotations.py

# 2. 可视化标注
python examples/visualize-labels.py

# 3. 人工抽查 10 张图片
```

### 练习 4：创建配置文件（15 分钟）
```bash
# 运行生成脚本
python examples/create-data-yaml.py

# 检查生成的 data.yaml
cat datasets/labeled/data.yaml
```

---

## ✅ 标注完成检查清单

标注完成后，确保：

- [ ] 所有图片都有标注
- [ ] 标注框紧贴目标边缘
- [ ] 没有漏标小目标
- [ ] 类别选择正确
- [ ] 标注值都在 0-1 范围内
- [ ] data.yaml 文件正确
- [ ] 可视化检查结果正常

---

## 🔗 相关资源

- [LabelImg GitHub](https://github.com/heartexlabs/labelImg)
- [LabelImg 教程](https://github.com/heartexlabs/labelImg#readme)
- [Roboflow 标注工具](https://roboflow.com/annotate)
- [CVAT 在线标注](https://cvat.ai/)

---

**下一步：[10-Model-Training - 模型训练](../10-model-training/README.md)** 🚀
