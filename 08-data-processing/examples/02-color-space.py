# examples/02-color-space.py
"""
色彩空间转换
"""
import cv2

# BGR 转 RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# BGR 转 HSV（用于颜色分割）
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# BGR 转灰度
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

print("色彩空间转换示例")
