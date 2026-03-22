# examples/03-photo-organizer.py
"""
批量拍摄后整理脚本

功能：整理拍摄的照片，删除模糊照片，按日期分类
"""
import os
from datetime import datetime
import shutil

def organize_photos(source_dir, output_dir):
    """
    整理拍摄的照片

    - 删除模糊照片
    - 按日期分类
    - 重命名
    """
    print(f"整理照片：{source_dir} -> {output_dir}")

if __name__ == "__main__":
    organize_photos("photos/raw/", "datasets/photos/")
