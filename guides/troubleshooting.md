# Troubleshooting - 常见问题与解决方法

> 学习过程中遇到的问题，这里可能有答案

---

## 🔧 环境配置问题

### Q1: PyTorch 安装失败

**错误信息：**
```
ERROR: Could not find a version that satisfies the requirement torch
```

**解决方法：**

```bash
# 1. 检查 Python 版本
python --version  # 需要 3.8-3.11

# 2. 访问 PyTorch 官网选择对应版本
# https://pytorch.org/get-started/locally/

# 3. 使用国内镜像
pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### Q2: CUDA 不可用

**错误信息：**
```
CUDA unavailable: False
```

**解决方法：**

```bash
# 1. 检查 NVIDIA 驱动
nvidia-smi  # 能看到 GPU 信息

# 2. 检查 CUDA 版本
nvcc --version

# 3. 重装带 CUDA 支持的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**临时方案：** 使用 CPU 训练（慢，但能跑）
```python
model.train(device='cpu')
```

---

### Q3: Ultralytics 安装失败

**错误信息：**
```
ERROR: Failed building wheel for xxx
```

**解决方法：**

```bash
# 1. 更新 pip
pip install --upgrade pip

# 2. 安装构建工具
# macOS
xcode-select --install
# Linux
sudo apt-get update && sudo apt-get install -y build-essential

# 3. 重装 ultralytics
pip install --no-cache-dir ultralytics
```

---

## 📊 数据问题

### Q4: 标注框位置不对

**现象：** 训练时框乱飞，或者推理结果偏移

**可能原因：**
1. 标注未归一化
2. 图片尺寸不一致
3. XML 转 YOLO 出错

**检查方法：**

```python
# 检查标注值是否在 [0, 1] 范围内
with open("labels/train/001.txt", 'r') as f:
    for line in f:
        values = [float(v) for v in line.split()[1:]]
        print(f"Min: {min(values):.4f}, Max: {max(values):.4f}")
        # 应该都在 0-1 之间
```

**解决方法：**
- 重新运行格式转换脚本
- 确保所有标注值归一化

---

### Q5: 训练时报错 "no labels"

**错误信息：**
```
WARNING: No labels found in /path/to/labels/train
```

**可能原因：**
1. 标注文件路径不对
2. 标注文件格式错误
3. 图片和标注文件名不匹配

**解决方法：**

```bash
# 1. 检查目录结构
tree datasets/

# 2. 检查文件名是否匹配
ls images/train/
ls labels/train/
# 应该一一对应

# 3. 运行质量检查
python examples/12-final-quality-check.py
```

---

### Q6: 类别不平衡

**现象：** 某些类别检测效果很差

**原因：** 某类数据太少

**解决方法：**

1. **针对性收集数据**
   ```bash
   # 针对性搜索
   googleimagesdownload --keywords "PCB short circuit" --limit 50
   ```

2. **过采样**
   ```python
   # 对少类别的图片多增强几倍
   ```

3. **修改 loss 权重**（进阶）
   ```yaml
   # 在训练配置中添加
   loss_ow: [1.0, 1.0, 3.0, 3.0, 2.0]  # 给少类别更高权重
   ```

---

## 🏋️ 训练问题

### Q7: 训练 loss 不下降

**现象：** loss 一直在某个值附近波动

**可能原因及解决：**

| 原因 | 检查方法 | 解决方法 |
|------|----------|----------|
| 学习率太高 | loss 波动大 | 降低 lr0 到 0.001 |
| 标注错误 | 可视化标注 | 重新标注 |
| 数据太少 | 统计各类数量 | 增加数据或增强 |
| 模型太小 | 换大模型试试 | yolov8n → yolov8s |

**调试步骤：**

```bash
# 1. 从小模型开始
yolo train model=yolov8n.pt data=data.yaml epochs=10

# 2. 检查训练可视化
tensorboard --logdir=runs/detect/train

# 3. 查看预测结果
yolo predict model=runs/detect/train/weights/best.pt source=datasets/val/images/
```

---

### Q8: 训练很慢

**现象：** 每个 epoch 要几十分钟

**可能原因：**

| 原因 | 解决方法 |
|------|----------|
| CPU 训练 | 用 GPU |
| batch_size 太大 | 减小到 8 或 16 |
| 图片太大 | 减小 imgsz 到 416 |
| workers 太多/太少 | 调整为 CPU 核心数 |

**优化配置：**

```python
model.train(
    data='data.yaml',
    batch=16,      # 减小批次
    imgsz=416,     # 减小图片尺寸
    workers=4,     # 数据加载线程数
    device=0       # 使用 GPU
)
```

---

### Q9: CUDA Out of Memory

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方法：**

```python
# 1. 减小 batch size
model.train(batch=8)  # 从 16 减到 8

# 2. 减小图片尺寸
model.train(imgsz=416)  # 从 640 减到 416

# 3. 使用混合精度训练
model.train(amp=True)

# 4. 换小模型
model = YOLO('yolov8n.pt')  # 用 nano 而不是 small
```

---

### Q10: 过拟合

**现象：** 训练集 mAP 很高，验证集 mAP 很低

**判断方法：**
```
train mAP@50: 0.95  ← 很高
val   mAP@50: 0.65  ← 低，差距大
```

**解决方法：**

1. **增加数据增强**
   ```python
   augment=True,  # 确保开启增强
   ```

2. **添加 dropout**（如果模型支持）

3. **早停**
   ```python
   patience=50,  # 50 轮不改进就停止
   ```

4. **减小模型**
   ```python
   model = YOLO('yolov8n.pt')  # 用更小的模型
   ```

---

### Q11: 欠拟合

**现象：** 训练集和验证集 mAP 都很低

**解决方法：**

1. **增加训练轮数**
   ```python
   epochs=300  # 从 100 增加到 300
   ```

2. **换大模型**
   ```python
   model = YOLO('yolov8m.pt')  # 用 medium 而不是 nano
   ```

3. **降低学习率**
   ```python
   lr0=0.001  # 从 0.01 降低
   ```

4. **检查数据质量**
   - 标注是否准确
   - 数据是否足够

---

## 📈 评估问题

### Q12: mAP 很低（<0.3）

**可能原因：**

| 原因 | 概率 | 解决方法 |
|------|------|----------|
| 训练轮数不够 | 高 | 增加 epochs |
| 数据质量差 | 高 | 重新检查标注 |
| 模型太小 | 中 | 换大模型 |
| 学习率不合适 | 中 | 调整 lr0 |

**调试流程：**

```
1. 先检查标注质量
   ↓
2. 增加训练轮数到 300
   ↓
3. 如果还不行，换大模型
   ↓
4. 最后调整超参数
```

---

### Q13: 某些类别检测不到

**现象：** 混淆矩阵显示某类全是漏检

**原因：**
1. 该类数据太少
2. 该类特征不明显
3. 标注不一致

**解决方法：**

```bash
# 1. 统计各类数量
python examples/class_distribution.py

# 2. 针对性增强该类数据
# 3. 检查该类标注质量
```

---

## 🚀 部署问题

### Q14: ONNX 推理结果不对

**现象：** PyTorch 模型正常，ONNX 推理结果差

**可能原因：**
1. 预处理不一致
2. 输入尺寸不匹配
3. 后处理有问题

**解决方法：**

```python
# 1. 确保导出时参数正确
model.export(format='onnx', dynamic=False, simplify=True)

# 2. 检查输入预处理
# ONNX 期望 RGB 归一化到 [0,1]

# 3. 使用官方推理代码对比
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
```

---

### Q15: 推理速度慢

**可能原因：**

| 原因 | 解决方法 |
|------|----------|
| CPU 推理 | 用 GPU 或 TensorRT |
| 模型太大 | 换小模型 |
| 图片太大 | 减小推理尺寸 |
| 批量太小 | 增大批次 |

**优化方案：**

```python
# 1. 使用 TensorRT（NVIDIA GPU）
model.export(format='engine', device=0)

# 2. 减小推理尺寸
results = model.predict(source=0, imgsz=416)

# 3. 批量推理
results = model.predict(source='images/', batch=8)
```

---

## 📱 其他问题

### Q16: 中文路径/文件名问题

**错误信息：**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte...
```

**解决方法：**

```bash
# 1. 避免中文路径
mv 数据集/数据集 1/ datasets/dataset1/

# 2. 如果必须用中文，确保系统支持
# Windows: 区域设置改为中文
# Linux: export LANG=zh_CN.UTF-8
```

---

### Q17: 导入错误

**错误信息：**
```
ImportError: No module named 'xxx'
```

**解决方法：**

```bash
# 1. 确认在当前环境安装
pip list | grep xxx

# 2. 重新安装
pip install xxx

# 3. 如果是 Jupyter，重启 kernel
```

---

## 🆘 还是解决不了？

### 提问的艺术

在 Issues 或论坛提问时，提供以下信息：

```markdown
**问题描述：**
简短描述你遇到的问题

**复现步骤：**
1. 运行了什麼命令
2. 看到了什么错误

**环境信息：**
- Python 版本：3.x.x
- PyTorch 版本：x.x.x
- 系统：Windows/macOS/Linux
- GPU: 有/无，型号

**错误日志：**
```
粘贴完整的错误信息
```

**已尝试的解决方法：**
1. 试过方法 A，结果...
2. 试过方法 B，结果...
```

---

## 📚 更多资源

- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [YOLOv5 Issues](https://github.com/ultralytics/yolov5/issues)
- [Stack Overflow - YOLO 标签](https://stackoverflow.com/questions/tagged/yolo)
- [PyTorch 论坛](https://discuss.pytorch.org/)

---

**遇到问题不要慌，99% 的问题都有人遇到过！** 💪
