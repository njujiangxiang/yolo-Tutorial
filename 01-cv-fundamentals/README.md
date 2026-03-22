# 01-CV-Fundamentals - 计算机视觉基础

> 学习计算机视觉和图像处理的基础知识

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 理解数字图像的基本概念
- ✅ 使用 OpenCV 读取和处理图像
- ✅ 掌握图像预处理技术
- ✅ 提取图像特征

---

## 📚 1. 数字图像基础

### 1.1 图像的表示

**灰度图像**：单通道，每个像素值 0-255

```
像素值 0 = 黑色
像素值 255 = 白色
```

**彩色图像**：三通道（RGB 或 BGR）

```
每个像素由 (R, G, B) 三个值组成
例如：(255, 0, 0) = 红色
```

### 1.2 图像基本属性

| 属性 | 说明 | 示例 |
|------|------|------|
| 分辨率 | 图像的宽高（像素） | 640x480 |
| 位深度 | 每个像素的位数 | 8 位 = 256 级 |
| 通道数 | 灰度 1 通道，彩色 3 通道 | RGB 三通道 |

---

## 🛠️ 2. OpenCV 基础

### 2.1 安装和导入

```bash
pip install opencv-python
```

```python
import cv2
import numpy as np
```

### 2.2 读取和显示图像

```python
# examples/01-read-image.py
import cv2

# 读取图像
img = cv2.imread('image.jpg')

# 显示图像
cv2.imshow('Image', img)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭窗口

# 保存图像
cv2.imwrite('output.jpg', img)
```

### 2.3 图像基本操作

```python
# examples/02-image-basic.py
import cv2

img = cv2.imread('image.jpg')

# 获取图像属性
print(f"形状：{img.shape}")  # (高，宽，通道数)
print(f"数据类型：{img.dtype}")

# 裁剪图像
cropped = img[100:300, 200:400]

# 调整大小
resized = cv2.resize(img, (320, 240))

# 旋转图像
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 旋转 45 度
rotated = cv2.warpAffine(img, M, (w, h))
```

---

## 📊 3. 图像预处理

### 3.1 灰度转换

```python
# examples/03-grayscale.py
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray.jpg', gray)
```

### 3.2 图像平滑（去噪）

```python
# examples/04-blur.py
import cv2

img = cv2.imread('image.jpg')

# 高斯模糊
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# 中值滤波
median = cv2.medianBlur(img, 5)

# 双边滤波（保边去噪）
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

### 3.3 边缘检测

```python
# examples/05-edge-detection.py
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny 边缘检测
edges = cv2.Canny(gray, 100, 200)

# Sobel 边缘检测
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
```

### 3.4 阈值处理

```python
# examples/06-threshold.py
import cv2

gray = cv2.imread('image.jpg', 0)

# 二值化
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 自适应阈值
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
```

### 3.5 形态学操作

```python
# examples/07-morphology.py
import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)
kernel = np.ones((5, 5), np.uint8)

# 腐蚀（消除小白点）
eroded = cv2.erode(img, kernel, iterations=1)

# 膨胀（消除小黑点）
dilated = cv2.dilate(img, kernel, iterations=1)

# 开运算（先腐蚀后膨胀）
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 闭运算（先膨胀后腐蚀）
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

---

## 🔍 4. 特征提取

### 4.1 轮廓检测

```python
# examples/08-contours.py
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# 绘制轮廓
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# 计算轮廓属性
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area > 100:  # 过滤小轮廓
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

### 4.2 特征点检测

```python
# examples/09-features.py
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB 特征点检测
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

# 绘制特征点
img_with_keypoints = cv2.drawKeypoints(
    gray, keypoints, None, color=(0, 255, 0)
)
```

---

## 💻 实战练习

### 练习 1：图像基本操作（30 分钟）

```python
# exercises/01-image-basic.py
"""
读取一张图片，完成以下操作：
1. 显示并保存为灰度图
2. 调整大小到原来的一半
3. 旋转 90 度
4. 裁剪中心区域
"""
import cv2

# TODO: 完成练习
```

### 练习 2：图像预处理（30 分钟）

```python
# exercises/02-preprocessing.py
"""
对图片进行预处理：
1. 高斯模糊去噪
2. Canny 边缘检测
3. 阈值二值化
4. 形态学开运算
"""
import cv2

# TODO: 完成练习
```

### 练习 3：缺陷检测初探（60 分钟）

```python
# exercises/03-defect-detection-intro.py
"""
使用传统 CV 方法检测简单缺陷：

场景：检测白色背景上的黑色划痕

步骤:
1. 读取图片
2. 转换为灰度
3. 阈值分割
4. 查找轮廓
5. 标记疑似缺陷区域
"""
import cv2
import numpy as np

def detect_scratch(image_path):
    """检测划痕"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # TODO: 完成缺陷检测逻辑
    pass

# TODO: 完成练习
```

---

## ✅ 学习检查清单

完成本章后，确保你能够：

- [ ] 使用 OpenCV 读取和保存图像
- [ ] 转换图像颜色空间（BGR ↔ Gray）
- [ ] 调整图像大小和旋转
- [ ] 应用模糊滤波器
- [ ] 执行边缘检测
- [ ] 进行阈值处理
- [ ] 查找和绘制轮廓
- [ ] 检测特征点

---

## 🔗 相关资源

- [OpenCV 官方文档](https://docs.opencv.org/)
- [OpenCV Python 教程](https://opencv-python-tutroals.readthedocs.io/)
- [计算机视觉入门](https://www.bilibili.com/video/BV1py4y1K7BT)

---

**下一步：[02-Deep-Learning - 深度学习基础](../02-deep-learning/README.md)** 🚀
