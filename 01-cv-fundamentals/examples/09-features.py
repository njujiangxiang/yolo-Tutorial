# examples/09-features.py
"""
特征点检测

学习使用 ORB 算法检测和匹配特征点
"""
import cv2

def main():
    # 读取图像
    img = cv2.imread('test.jpg')

    if img is None:
        print("无法读取图像")
        return

    # 转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB 特征点检测
    orb = cv2.ORB_create(nfeatures=500)

    # 检测关键点和计算描述子
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    print(f"检测到 {len(keypoints)} 个特征点")

    # 绘制特征点
    img_with_keypoints = cv2.drawKeypoints(
        gray, keypoints, None, color=(0, 255, 0), flags=0
    )
    cv2.imwrite('features_orb.jpg', img_with_keypoints)
    print("已保存 ORB 特征点图像：features_orb.jpg")

    # 绘制带方向的关键点
    img_with_directions = cv2.drawKeypoints(
        gray, keypoints, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite('features_directions.jpg', img_with_directions)
    print("已保存带方向的特征点图像：features_directions.jpg")

    print("\n提示:")
    print("- ORB: 快速，适合实时应用")
    print("- SIFT: 精度高，但有专利限制")
    print("- SURF: 速度快，但有专利限制")
    print("- 特征点用于：图像匹配、目标识别、全景拼接")

if __name__ == "__main__":
    main()
