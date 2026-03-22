# examples/08-contours.py
"""
轮廓检测

学习查找和绘制图像轮廓
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

    # 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"找到 {len(contours)} 个轮廓")

    # 绘制所有轮廓
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite('contours_all.jpg', contour_img)
    print("已保存所有轮廓图像：contours_all.jpg")

    # 分析每个轮廓
    result_img = img.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # 过滤小轮廓
        if area < 100:
            continue

        # 计算周长
        perimeter = cv2.arcLength(contour, True)

        # 外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 绘制矩形
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 标注面积
        cv2.putText(result_img, f'{area:.0f}', (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite('contours_filtered.jpg', result_img)
    print("已保存过滤后轮廓图像：contours_filtered.jpg")

    print("\n轮廓属性:")
    print("- area: 面积")
    print("- perimeter: 周长")
    print("- boundingRect: 外接矩形")
    print("- minEnclosingCircle: 外接圆")
    print("- fitEllipse: 拟合椭圆")

if __name__ == "__main__":
    main()
