# examples/04-remove-duplicates.py
"""
删除重复图片

使用感知哈希算法
"""
import os
from pathlib import Path
from PIL import Image
import imagehash

def find_duplicate_images(input_dir, threshold=0.95):
    """
    查找重复图片（使用感知哈希）

    返回重复图片列表
    """
    hashes = {}
    duplicates = []

    for img_path in Path(input_dir).glob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                # 计算感知哈希
                phash = imagehash.phash(img)

                # 检查是否与已有图片重复
                for existing_path, existing_hash in hashes.items():
                    similarity = 1 - (phash - existing_hash) / len(phash) ** 2

                    if similarity >= threshold:
                        duplicates.append((str(img_path), str(existing_path), similarity))
                        break

                if not duplicates or duplicates[-1][0] != str(img_path):
                    hashes[str(img_path)] = phash

        except Exception as e:
            print(f"处理失败 {img_path}: {e}")

    return duplicates

def remove_duplicates(input_dir, output_dir):
    """删除重复图片，保留一份"""
    os.makedirs(output_dir, exist_ok=True)

    duplicates = find_duplicate_images(input_dir)
    removed_paths = set([d[0] for d in duplicates])

    kept = 0
    for img_path in Path(input_dir).glob("*.jpg"):
        if str(img_path) not in removed_paths:
            os.system(f"cp {img_path} {output_dir}/")
            kept += 1

    print(f"删除 {len(removed_paths)} 张重复图片，保留 {kept} 张")

if __name__ == "__main__":
    # 先安装：pip install Pillow imagehash
    remove_duplicates("datasets/cleaned/", "datasets/deduped/")
