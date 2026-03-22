# examples/test-api.py
"""
测试 FastAPI 服务

使用 Python 请求测试 Web 服务
"""
import requests

# 上传图片
with open('test.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

# 打印结果
print("响应状态码:", response.status_code)
print("检测结果:", response.json())
