# exercises/01-iou-loss.py
"""
不同 IoU 损失变体

理解 IoU、GIoU、DIoU、CIoU 损失的计算和作用
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calculate_iou(box1, box2):
    """计算 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_giou(box1, box2):
    """
    计算 GIoU (Generalized IoU)
    解决 IoU 为 0 时梯度消失问题
    """
    iou = calculate_iou(box1, box2)

    # 最小外接矩形
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    C_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - calculate_iou(box1, box2) * (area1 + area2) / (1 + calculate_iou(box1, box2))

    giou = iou - (C_area - union) / C_area

    return giou


def calculate_diou(box1, box2):
    """
    计算 DIoU (Distance IoU)
    考虑中心点距离
    """
    iou = calculate_iou(box1, box2)

    # 中心点距离
    c1_x = (box1[0] + box1[2]) / 2
    c1_y = (box1[1] + box1[3]) / 2
    c2_x = (box2[0] + box2[2]) / 2
    c2_y = (box2[1] + box2[3]) / 2

    center_dist = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    # 对角线距离
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    diagonal = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    diou = iou - (center_dist / diagonal) ** 2

    return diou


def calculate_ciou(box1, box2):
    """
    计算 CIoU (Complete IoU)
    考虑重叠、距离、长宽比
    """
    iou = calculate_iou(box1, box2)

    # 中心点距离
    c1_x = (box1[0] + box1[2]) / 2
    c1_y = (box1[1] + box1[3]) / 2
    c2_x = (box2[0] + box2[2]) / 2
    c2_y = (box2[1] + box2[3]) / 2

    center_dist = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    # 对角线距离
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    diagonal = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 长宽比一致性
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    v = (4 / (np.pi ** 2)) * (np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2

    alpha = v / (1 - iou + v) if (1 - iou + v) > 0 else 0

    ciou = iou - (center_dist / diagonal) ** 2 - alpha * v

    return ciou


def visualize_boxes(boxes, title):
    """可视化边界框"""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['red', 'blue', 'green', 'orange']

    for i, box in enumerate(boxes):
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor='none',
            label=f'Box {i}'
        )
        ax.add_patch(rect)

        # 标注中心点
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        ax.plot(cx, cy, 'o', color=colors[i % len(colors)], markersize=8)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    print(f"已保存：{title.replace(' ', '_')}.png")


def main():
    print("=" * 50)
    print("IoU 损失变体比较")
    print("=" * 50)

    # 测试用例 1：部分重叠
    box1 = [100, 100, 200, 200]
    box2 = [150, 150, 250, 250]

    print("\n测试 1: 部分重叠")
    print(f"  Box1: {box1}")
    print(f"  Box2: {box2}")

    iou = calculate_iou(box1, box2)
    giou = calculate_giou(box1, box2)
    diou = calculate_diou(box1, box2)
    ciou = calculate_ciou(box1, box2)

    print(f"  IoU:  {iou:.4f}")
    print(f"  GIoU: {giou:.4f}")
    print(f"  DIoU: {diou:.4f}")
    print(f"  CIoU: {ciou:.4f}")

    # 测试用例 2：不重叠
    box3 = [50, 50, 100, 100]
    box4 = [250, 250, 300, 300]

    print("\n测试 2: 不重叠")
    print(f"  Box3: {box3}")
    print(f"  Box4: {box4}")

    iou = calculate_iou(box3, box4)
    giou = calculate_giou(box3, box4)
    diou = calculate_diou(box3, box4)
    ciou = calculate_ciou(box3, box4)

    print(f"  IoU:  {iou:.4f}")
    print(f"  GIoU: {giou:.4f}")
    print(f"  DIoU: {diou:.4f}")
    print(f"  CIoU: {ciou:.4f}")

    # 测试用例 3：包含关系
    box5 = [100, 100, 300, 300]
    box6 = [150, 150, 250, 250]

    print("\n测试 3: 包含关系")
    print(f"  Box5 (外): {box5}")
    print(f"  Box6 (内): {box6}")

    iou = calculate_iou(box5, box6)
    giou = calculate_giou(box5, box6)
    diou = calculate_diou(box5, box6)
    ciou = calculate_ciou(box5, box6)

    print(f"  IoU:  {iou:.4f}")
    print(f"  GIoU: {giou:.4f}")
    print(f"  DIoU: {diou:.4f}")
    print(f"  CIoU: {ciou:.4f}")

    # 可视化
    visualize_boxes([box1, box2], "部分重叠")
    visualize_boxes([box3, box4], "不重叠")
    visualize_boxes([box5, box6], "包含关系")

    print("\n" + "=" * 50)
    print("IoU 损失解读:")
    print("=" * 50)
    print("IoU:  基础指标，但不重叠时梯度消失")
    print("GIoU: 解决梯度消失，但收敛较慢")
    print("DIoU: 考虑中心距离，收敛更快")
    print("CIoU: 考虑重叠 + 距离 + 长宽比，最全面")
    print("\nYOLOv8 使用：DFL + CIoU 组合损失")


if __name__ == "__main__":
    main()
