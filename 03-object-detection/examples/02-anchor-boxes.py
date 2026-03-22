# examples/02-anchor-boxes.py
"""
锚框可视化演示

理解 YOLO 中锚框的概念和作用
"""
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("=" * 50)
    print("YOLO 锚框可视化")
    print("=" * 50)

    # YOLOv3 的锚框（归一化前）
    # 分为三组，对应三个检测尺度
    anchors = [
        # 小目标锚框
        ([10, 13], "小"),
        ([16, 30], "小"),
        ([33, 23], "小"),
        # 中目标锚框
        ([30, 61], "中"),
        ([62, 45], "中"),
        ([59, 119], "中"),
        # 大目标锚框
        ([116, 90], "大"),
        ([156, 198], "大"),
        ([373, 326], "大"),
    ]

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = {
        "小": "green",
        "中": "blue",
        "大": "red"
    }

    # 绘制每个锚框
    for (w, h), size in anchors:
        # 以原点为中心绘制矩形
        rect = plt.Rectangle(
            (-w/2, -h/2), w, h,
            fill=False, linewidth=2,
            color=colors[size],
            label=f'{size}目标：{w}x{h}'
        )
        ax.add_patch(rect)

    # 设置图表
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('YOLO Anchor Boxes (锚框)')
    ax.grid(True, alpha=0.3)

    # 添加说明
    plt.figtext(0.02, 0.02,
                '绿色：小目标锚框\n'
                '蓝色：中目标锚框\n'
                '红色：大目标锚框',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig('anchor_boxes.png', dpi=150, bbox_inches='tight')
    print("\n锚框图已保存：anchor_boxes.png")

    print("\n锚框说明:")
    print("=" * 50)
    print("1. 锚框是预定义的边界框尺寸")
    print("2. YOLO 在每个位置预设多个锚框")
    print("3. 训练时选择最匹配的锚框进行预测")
    print("4. 多尺度锚框可以检测不同大小的目标")
    print("\nYOLOv3 锚框示例:")
    for (w, h), size in anchors:
        print(f"  {size}目标：{w}x{h}")


if __name__ == "__main__":
    main()
