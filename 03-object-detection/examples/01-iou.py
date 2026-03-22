# examples/01-iou.py
"""
计算 IoU（交并比）

理解目标检测中最基础的评估指标
"""

def calculate_iou(box1, box2):
    """
    计算两个框的 IoU

    参数:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    返回:
        IoU 值 (0-1)
    """
    # 计算交集坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union = area1 + area2 - intersection

    # 计算 IoU
    iou = intersection / union if union > 0 else 0

    return iou


def main():
    print("=" * 50)
    print("IoU（交并比）计算")
    print("=" * 50)

    # 测试用例 1：部分重叠
    box_a = [50, 50, 150, 150]
    box_b = [100, 100, 200, 200]
    iou1 = calculate_iou(box_a, box_b)
    print(f"\n测试 1: 部分重叠")
    print(f"  框 A: {box_a}")
    print(f"  框 B: {box_b}")
    print(f"  IoU: {iou1:.4f}")

    # 测试用例 2：完全重合
    box_c = [100, 100, 200, 200]
    box_d = [100, 100, 200, 200]
    iou2 = calculate_iou(box_c, box_d)
    print(f"\n测试 2: 完全重合")
    print(f"  框 C: {box_c}")
    print(f"  框 D: {box_d}")
    print(f"  IoU: {iou2:.4f} (完美)")

    # 测试用例 3：不重叠
    box_e = [50, 50, 100, 100]
    box_f = [200, 200, 250, 250]
    iou3 = calculate_iou(box_e, box_f)
    print(f"\n测试 3: 不重叠")
    print(f"  框 E: {box_e}")
    print(f"  框 F: {box_f}")
    print(f"  IoU: {iou3:.4f} (无重叠)")

    # 测试用例 4：包含关系
    box_g = [50, 50, 200, 200]
    box_h = [100, 100, 150, 150]
    iou4 = calculate_iou(box_g, box_h)
    print(f"\n测试 4: 包含关系")
    print(f"  框 G (外): {box_g}")
    print(f"  框 H (内): {box_h}")
    print(f"  IoU: {iou4:.4f}")

    print("\n" + "=" * 50)
    print("IoU 解读:")
    print("=" * 50)
    print("IoU = 1.0: 完美重合")
    print("IoU ≥ 0.5: 通常认为是正确检测")
    print("IoU ≥ 0.75: 严格正确")
    print("IoU = 0.0: 无重叠")


if __name__ == "__main__":
    main()
