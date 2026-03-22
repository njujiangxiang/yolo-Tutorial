# examples/04-metrics.py
"""
目标检测评估指标

计算 Precision, Recall, mAP 等指标
"""
import numpy as np


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


def evaluate_detection(ground_truth, predictions, iou_threshold=0.5):
    """
    评估目标检测结果

    参数:
        ground_truth: 真实标注列表
        predictions: 预测结果列表
        iou_threshold: IoU 阈值

    返回:
        TP, FP, FN, Precision, Recall
    """
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = 0  # 假反例

    matched_gt = set()  # 已匹配的真实框

    for pred in predictions:
        pred_box = pred['box']
        best_iou = 0
        best_gt_idx = -1

        # 找到最匹配的真实框
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # 判断 TP 或 FP
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    # 计算 FN（未匹配的真实框）
    fn = len(ground_truth) - len(matched_gt)

    # 计算 Precision 和 Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    }


def main():
    print("=" * 50)
    print("目标检测评估指标")
    print("=" * 50)

    # 真实标注（3 个缺陷）
    ground_truth = [
        {'box': [50, 50, 100, 100], 'class': 'defect'},
        {'box': [200, 200, 250, 250], 'class': 'defect'},
        {'box': [300, 300, 350, 350], 'class': 'defect'},
    ]

    # 模型预测（4 个检测）
    predictions = [
        {'box': [55, 55, 105, 105], 'conf': 0.9, 'class': 'defect'},  # TP
        {'box': [210, 210, 260, 260], 'conf': 0.8, 'class': 'defect'},  # TP
        {'box': [400, 400, 450, 450], 'conf': 0.7, 'class': 'defect'},  # FP (无对应真实框)
        {'box': [305, 305, 355, 355], 'conf': 0.6, 'class': 'defect'},  # TP
    ]

    print("\n真实标注:")
    for i, gt in enumerate(ground_truth):
        print(f"  GT{i}: {gt['box']}")

    print("\n模型预测:")
    for i, pred in enumerate(predictions):
        print(f"  Pred{i}: {pred['box']}, conf={pred['conf']:.2f}")

    # 评估
    results = evaluate_detection(ground_truth, predictions, iou_threshold=0.5)

    print("\n" + "=" * 50)
    print("评估结果:")
    print("=" * 50)
    print(f"TP (真正例): {results['TP']}")
    print(f"FP (假正例): {results['FP']}")
    print(f"FN (假反例): {results['FN']}")
    print("-" * 50)
    print(f"Precision: {results['Precision']:.2%}")
    print(f"Recall:    {results['Recall']:.2%}")
    print(f"F1 Score:  {results['F1']:.2f}")

    print("\n" + "=" * 50)
    print("指标解读:")
    print("=" * 50)
    print("TP: 真有缺陷，检测到了 ✓")
    print("FP: 没有缺陷，误报了 ✗")
    print("FN: 有缺陷，没检测到 ✗ (最危险!)")
    print("\nPrecision: 检测出的缺陷中有多少是真的")
    print("Recall: 所有真实缺陷中有多少被检测到了")
    print("F1 Score: Precision 和 Recall 的调和平均")

    print("\n" + "=" * 50)
    print("场景选择:")
    print("=" * 50)
    print("安全检测 → 优先保证 Recall (宁可误报，不可漏检)")
    print("质量分选 → 优先保证 Precision (避免误判)")


if __name__ == "__main__":
    main()
