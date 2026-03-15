# 05-YOLOv5 - YOLOv5 实战

> 学习使用 YOLOv5 进行目标检测

---

## 🎯 学习目标

- ✅ YOLOv5 架构
- ✅ 模型配置
- ✅ 推理使用
- ✅ 训练流程

---

## 📦 安装

```bash
pip install git+https://github.com/ultralytics/yolov5.git
```

---

## 💻 示例代码

### 1. 基础推理

```python
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 推理
img = 'image.jpg'
results = model(img)

# 显示结果
results.show()

# 获取结果
df = results.pandas().xyxy[0]
print(df)
```

### 2. 批量推理

```python
# 多张图片
imgs = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model(imgs)

# 视频
results = model('video.mp4')
```

### 3. 训练模型

```bash
# 使用自定义数据集
python train.py --data custom.yaml --weights yolov5s.pt --epochs 100
```

---

## 📝 练习

1. 使用 YOLOv5 检测图片
2. 保存检测结果
3. 处理视频文件
4. 尝试不同模型大小

---

**继续学习：[06-YOLOv8](../06-yolov8/README.md)** 🚀
