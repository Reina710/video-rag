# -*- coding: utf-8 -*-
"""测试：用 DashScope qwen-vl-ocr 识别一张关键帧。"""
import base64
import os
import sys

# 清代理
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

from openai import OpenAI

import settings

key = settings.get("DASHSCOPE_KEY")
if not key:
    raise SystemExit("没找到 DASHSCOPE_KEY（写在 .env 或环境变量里）")

client = OpenAI(
    api_key=key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "frames", "test_60s.jpg"
)
with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
data_url = "data:image/jpeg;base64," + b64

resp = client.chat.completions.create(
    model="qwen-vl-ocr",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "请仅输出图像中的文本内容。"},
        ],
    }],
)
print("OCR 结果：")
print(resp.choices[0].message.content)
