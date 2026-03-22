# examples/01-web-scraper.py
"""
使用 Python 爬虫收集图片

功能：从搜索引擎下载指定关键词的图片
"""
import requests
from bs4 import BeautifulSoup
import os
import time

def download_images(query, num_images=50):
    """
    从搜索引擎下载图片

    参数:
        query: 搜索关键词
        num_images: 下载数量
    """
    # 创建保存目录
    save_dir = f"datasets/raw/{query}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"搜索关键词：{query}")
    print(f"保存目录：{save_dir}")
    print(f"计划下载：{num_images} 张图片")

    # 注意：实际使用需要调用搜索引擎 API
    # 这里只是示例结构
    # 建议使用：google-images-download 或 imglyb 库

    print("\n提示：实际使用时，请使用以下工具之一：")
    print("1. google-images-download 库")
    print("2. imglyb 库")
    print("3. 手动从公开数据集下载")

if __name__ == "__main__":
    download_images("PCB 缺陷", 50)
