# examples/01-read-image.py
"""
读取和显示图像

学习使用 OpenCV 读取、显示和保存图像
"""
import cv2

def main():
    # 读取图像
    img = cv2.imread('test.jpg')

    if img is None:
        print("无法读取图像，请确保 test.jpg 存在")
        return

    # 显示图像
    cv2.imshow('Image', img)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭窗口

    # 保存图像
    cv2.imwrite('output.jpg', img)
    print("图像已保存为 output.jpg")

    # 获取图像属性
    print(f"\n图像信息:")
    print(f"  形状：{img.shape} (高，宽，通道数)")
    print(f"  数据类型：{img.dtype}")
    print(f"  像素总数：{img.size}")

if __name__ == "__main__":
    main()
