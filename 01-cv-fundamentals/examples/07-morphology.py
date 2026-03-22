# examples/07-morphology.py
"""
形态学操作

学习使用形态学操作处理二值图像
"""
import cv2
import numpy as np

def main():
    # 读取图像（灰度）
    img = cv2.imread('test.jpg', 0)

    if img is None:
        print("无法读取图像")
        return

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 创建结构元素
    kernel = np.ones((5, 5), np.uint8)

    # 1. 腐蚀
    # 消除小白点，细化物体
    eroded = cv2.erode(binary, kernel, iterations=1)
    cv2.imwrite('morph_erode.jpg', eroded)
    print("已保存腐蚀图像：morph_erode.jpg")

    # 2. 膨胀
    # 消除小黑点，填充空洞
    dilated = cv2.dilate(binary, kernel, iterations=1)
    cv2.imwrite('morph_dilate.jpg', dilated)
    print("已保存膨胀图像：morph_dilate.jpg")

    # 3. 开运算（先腐蚀后膨胀）
    # 消除小物体，平滑边界
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('morph_opening.jpg', opening)
    print("已保存开运算图像：morph_opening.jpg")

    # 4. 闭运算（先膨胀后腐蚀）
    # 填充小空洞，连接邻近物体
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('morph_closing.jpg', closing)
    print("已保存闭运算图像：morph_closing.jpg")

    # 5. 形态学梯度
    # 提取边缘
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite('morph_gradient.jpg', gradient)
    print("已保存形态学梯度图像：morph_gradient.jpg")

    print("\n提示:")
    print("- 腐蚀：物体变小，消除亮点")
    print("- 膨胀：物体变大，消除暗点")
    print("- 开运算：消除小物体")
    print("- 闭运算：填充小空洞")
    print("- iterations: 操作重复次数")

if __name__ == "__main__":
    main()
