# examples/10-xml-to-yolo.py
"""
将 LabelImg XML 格式转换为 YOLO 格式

XML 格式：[xmin, ymin, xmax, ymax]
YOLO 格式：[x_center, y_center, width, height] 归一化
"""
import xml.etree.ElementTree as ET
import os
from pathlib import Path

def xml_to_yolo(xml_path, image_path, output_path):
    """
    将 LabelImg XML 转换为 YOLO 格式
    """
    # 读取 XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # 转换标注
    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')

        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 转换为 YOLO 格式
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # 保存
    with open(output_path, 'w') as f:
        f.writelines(yolo_lines)

def batch_xml_to_yolo(xml_dir, output_dir):
    """批量转换"""
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in Path(xml_dir).glob("*.xml"):
        xml_to_yolo(
            str(xml_file),
            f"{xml_dir}/{xml_file.stem}.jpg",
            f"{output_dir}/{xml_file.stem}.txt"
        )

    print(f"转换完成：{output_dir}")

if __name__ == "__main__":
    batch_xml_to_yolo("datasets/xml_labels/", "datasets/yolo_labels/")
