# -*- coding: utf-8 -*-
"""
OCR 关键帧管线：视频 -> 抽关键帧 -> dHash去重 -> qwen-vl-ocr识别画面文字。

输出 frames_ocr.json：[{start毫秒, text}]，和 sentences.json 对齐，
后面一起存进 Qdrant（多模态双分支：ASR语音 + OCR画面文字）。

和原项目 DOVideo-AI 的对应：
  原项目：select=gt(scene,0.35) 场景检测 + dHash(汉明距离<=5)去重 + Tesseract OCR
  本项目：每30秒定时抽帧（简单可靠）+ 同样的dHash去重 + qwen-vl-ocr（比Tesseract准，
          且传base64不走公网URL，绕开Paraformer那个死结）

面试可讲的点（[面试] 标注）：
  - 为什么抽帧而不是逐帧OCR：画面静止时文字不变，逐帧OCR纯浪费钱和时间
  - dHash 是什么：9x8缩略图比相邻像素大小生成64位指纹，汉明距离<=5判重复。
    感知哈希对压缩噪声/轻微缩放不敏感，比逐像素比较稳
  - 为什么选 qwen-vl-ocr 而不是 Tesseract：中文识别精度差一个量级，
    且不用装系统软件
"""
import base64
import json
import os
import subprocess
import sys

import settings

# ---------- 0. 清代理 ----------
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else ""
OUT_JSON = os.path.join(BASE, "frames_ocr.json")
FRAMES_DIR = os.path.join(BASE, "frames")
if not VIDEO_PATH:
    raise SystemExit("用法: python ocr_frames.py <视频路径>")

# ffmpeg 路径（imageio-ffmpeg 自带）
import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# 每 30 秒抽一帧
STEP_SEC = 30

# ---------- 1. dHash 感知哈希 ----------
def dhash(image, hash_size=8):
    """把图片缩成 9x8 灰度图，比较每行相邻像素明暗 -> 64 位指纹。"""
    from PIL import Image
    img = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(img.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            # 左边像素比右边暗 -> 1，否则 0
            bits.append(1 if pixels[row * (hash_size + 1) + col] > pixels[row * (hash_size + 1) + col + 1] else 0)
    return bits

def hamming(a, b):
    """汉明距离：两个指纹有几位不一样。"""
    return sum(x != y for x, y in zip(a, b))

# ---------- 2. 抽帧（每 30 秒一帧） ----------
os.makedirs(FRAMES_DIR, exist_ok=True)
print("开始抽帧，每 %d 秒一帧..." % STEP_SEC)

# 先拿视频总时长
probe = subprocess.run(
    [FFMPEG, "-i", VIDEO_PATH],
    capture_output=True, text=True, encoding="utf-8", errors="ignore",
)
duration = 0
for line in probe.stderr.splitlines():
    if "Duration:" in line:
        h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
        duration = int(h) * 3600 + int(m) * 60 + float(s)
        break
print("视频时长: %.1f 秒" % duration)

# 逐帧抽取
import io
from PIL import Image

frame_files = []  # (时间秒, 文件路径)
t = 0
idx = 0
while t < duration:
    out = os.path.join(FRAMES_DIR, "f_%04d.jpg" % idx)
    r = subprocess.run(
        [FFMPEG, "-ss", str(t), "-i", VIDEO_PATH, "-frames:v", "1",
         "-q:v", "2", "-update", "1", "-y", out],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    if os.path.exists(out) and os.path.getsize(out) > 0:
        frame_files.append((t, out))
    t += STEP_SEC
    idx += 1

print("共抽出 %d 帧" % len(frame_files))

# ---------- 3. dHash 去重 ----------
print("dHash 去重中...")
kept = []          # [(秒, 路径)]
last_hash = None
for t, path in frame_files:
    img = Image.open(path)
    h = dhash(img)
    if last_hash is None or hamming(h, last_hash) > 5:
        kept.append((t, path))
        last_hash = h
    else:
        os.remove(path)  # 和上一帧几乎一样，删掉省 OCR 钱

print("去重后剩 %d 帧要 OCR" % len(kept))

# ---------- 4. OCR（qwen-vl-ocr） ----------
from openai import OpenAI

key = settings.get("DASHSCOPE_KEY")
if not key:
    raise SystemExit("没找到 DASHSCOPE_KEY（写在 .env 或环境变量里）")

ocr_client = OpenAI(
    api_key=key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def ocr_image(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = ocr_client.chat.completions.create(
        model="qwen-vl-ocr",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": "请仅输出图像中的文本内容。"},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()

results = []
for i, (t, path) in enumerate(kept):
    try:
        text = ocr_image(path)
    except Exception as e:
        print("  [%4d秒] OCR失败: %s" % (t, type(e).__name__))
        continue
    # 过滤：没识别出内容 / 只剩水印的帧不入库
    if not text or len(text) < 2:
        continue
    results.append({"start": t * 1000, "text": text})
    print("  [%02d:%02d] %s" % (t // 60, t % 60, text[:40].replace("\n", " / ")))

# ---------- 5. 落盘 ----------
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n共 %d 帧有文字，已写入 %s" % (len(results), OUT_JSON))
