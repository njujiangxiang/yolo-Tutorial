# examples/03-nms.py
"""
NMS（非极大值抑制）演示

理解如何去除重叠的检测框
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calculate_iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制

    参数:
        boxes: 边界框 [[x1,y1,x2,y2], ...]
        scores: 置信度分数
        iou_threshold: IoU 阈值

    返回:
        保留的框索引
    """
    if len(boxes) == 0:
        return []

    # 按分数排序（降序）
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        # 选择分数最高的框
        current = indices[0]
        keep.append(current)

        # 计算与其余框的 IoU
        if len(indices) > 1:
            ious = []
            for i in range(1, len(indices)):
                iou = calculate_iou(boxes[current], boxes[indices[i]])
                ious.append(iou)

            # 保留 IoU 低于阈值的框
            indices = indices[1:][np.array(ious) < iou_threshold]
        else:
            break

    return keep


def visualize_nms(boxes, scores, keep):
    """可视化 NMS 前后对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # NMS 前
    ax1.set_title('NMS 前')
    ax1.set_xlim(0, 400)
    ax1.set_ylim(0, 400)
    ax1.set_aspect('equal')

    for i, (box, score) in enumerate(zip(boxes, scores)):
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            alpha=0.6,
            label=f'Box {i}: {score:.2f}' if i < 3 else None
        )
        ax1.add_patch(rect)
        ax1.text(box[0], box[1] - 5, f'{score:.2f}',
                fontsize=9, color='red')

    ax1.grid(True, alpha=0.3)

    # NMS 后
    ax2.set_title(f'NMS 后 (保留 {len(keep)} 个框)')
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 400)
    ax2.set_aspect('equal')

    for i in keep:
        rect = patches.Rectangle(
            (boxes[i][0], boxes[i][1]),
            boxes[i][2] - boxes[i][0],
            boxes[i][3] - boxes[i][1],
            linewidth=2,
            edgecolor='green',
            facecolor='none',
            alpha=0.6
        )
        ax2.add_patch(rect)
        ax2.text(boxes[i][0], boxes[i][1] - 5,
                f'Box {i}: {scores[i]:.2f}',
                fontsize=9, color='green')

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nms_visualization.png', dpi=150)
    print("NMS 可视化已保存：nms_visualization.png")


def main():
    print("=" * 50)
    print("NMS（非极大值抑制）演示")
    print("=" * 50)

    # 示例数据：多个重叠的检测框
    boxes = [
        [50, 50, 150, 150],    # 框 0
        [55, 55, 155, 155],    # 框 1 (与框 0 重叠)
        [60, 60, 160, 160],    # 框 2 (与框 0 重叠)
        [200, 200, 300, 300],  # 框 3 (独立)
        [210, 210, 310, 310],  # 框 4 (与框 3 重叠)
        [350, 350, 400, 400],  # 框 5 (独立)
    ]

    scores = [0.90, 0.85, 0.80, 0.95, 0.75, 0.70]

    print("\n原始检测框:")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  框 {i}: {box}, 分数={score:.2f}")

    # 运行 NMS
    keep = nms(boxes, scores, iou_threshold=0.5)

    print(f"\nNMS 后保留的框:")
    for i in keep:
        print(f"  框 {i}: {boxes[i]}, 分数={scores[i]:.2f}")

    # 验证 IoU
    print(f"\n重叠框 IoU 验证:")
    iou_01 = calculate_iou(boxes[0], boxes[1])
    iou_34 = calculate_iou(boxes[3], boxes[4])
    print(f"  框 0 和框 1 的 IoU: {iou_01:.4f}")
    print(f"  框 3 和框 4 的 IoU: {iou_34:.4f}")

    # 可视化
    visualize_nms(boxes, scores, keep)

    print("\n" + "=" * 50)
    print("NMS 原理:")
    print("=" * 50)
    print("1. 按置信度分数排序")
    print("2. 选择分数最高的框")
    print("3. 移除与其 IoU 超过阈值的其他框")
    print("4. 重复步骤 2-3 直到没有框")
    print("\n作用：去除重复检测，保留最优结果")


if __name__ == "__main__":
    main()
