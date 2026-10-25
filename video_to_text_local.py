# -*- coding: utf-8 -*-
"""
第一步（本地版）：视频 -> 带时间戳的文字片段。

和旧版 video_to_text.py（Paraformer 云 API）的区别：
  旧版：把视频 URL 发给阿里云，要求视频有公网地址（本地视频够不着）
  新版：用 faster-whisper 在【本地】转写，视频就在磁盘上，彻底不需要公网 URL

faster-whisper 内部会用 ffmpeg 抽音频，所以前提是 ffmpeg 可用。

关键设计点：
  - 为什么不用 Paraformer 云 API：云 API 只认公网 URL，本地视频要额外"上公网"这一步
  - 为什么不用原版 openai-whisper：强依赖 PyTorch（2GB+），faster-whisper 用 CTranslate2
    量化加速，不需要 torch，速度快 4 倍，内存占用小
  - 为什么结果要带时间戳：后面问答"证据绑定"靠它
"""
import json
import sys
import os

import settings

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------- 0. 清理代理 + 设置 hf 国内镜像 ----------
# 环境里可能有坏的代理（HTTP_PROXY/HTTPS_PROXY 指向本地端口），
# 会导致 huggingface 下载模型返回 502。这里先清掉，再走国内镜像 hf-mirror。
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
# 用国内镜像下载 whisper 模型，绕开 huggingface 直连（国内慢/被墙）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------- 0. 把 ffmpeg 加进 PATH（faster-whisper 用 av 库解码，需要 ffmpeg） ----------
# imageio-ffmpeg 自带 ffmpeg.exe，但不在系统 PATH 里，这里手动把它的目录塞进去
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    print("已把 ffmpeg 目录加入 PATH:", ffmpeg_dir)
except ImportError:
    print("WARN: 没装 imageio-ffmpeg，尝试用系统 ffmpeg")

# ---------- 1. 你要转写的视频路径（命令行传入） ----------
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else ""
if not VIDEO_PATH:
    print("ERROR: 用法 python video_to_text_local.py <视频路径>")
    sys.exit(1)

# 本地模型路径：优先环境变量，其次旧的本地目录，最后让库自己下 small
MODEL_PATH = settings.get("WHISPER_MODEL_PATH") or (
    r"C:\Users\11651\tools\whisper-models\models"
    r"\pengzhendong--faster-whisper-small\snapshots\master"
)
if not os.path.isdir(MODEL_PATH):
    MODEL_PATH = "small"

# ---------- 2. 加载 faster-whisper ----------
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERROR: 没装 faster-whisper")
    print("安装命令：pip install faster-whisper")
    sys.exit(1)

if not os.path.exists(VIDEO_PATH):
    print("ERROR: 视频文件不存在:", VIDEO_PATH)
    sys.exit(1)

print("正在加载本地模型 %s ..." % MODEL_PATH)
# device="cpu"（本地没 GPU 用 CPU），compute_type 用 int8 加速省内存
# 直接用本地路径，不走 huggingface 下载
model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

# ---------- 3. 转写 ----------
print("开始转写:", VIDEO_PATH)
# 先关掉 vad_filter（语音活动检测），避免它把轻人声/短语音误杀
# beam_size=5 提高准确率
segments, info = model.transcribe(
    VIDEO_PATH,
    language="zh",
    vad_filter=False,
    beam_size=5,
    condition_on_previous_text=False,
)

print("检测语言: %s (置信度 %.2f)" % (info.language, info.language_probability))

# ---------- 4. 整理成 [{start, end, text}]，毫秒单位（和旧版对齐） ----------
sentences = []
for seg in segments:
    # faster-whisper 的 start/end 是【秒】的浮点数，转成毫秒整数
    sentences.append({
        "start": int(seg.start * 1000),
        "end": int(seg.end * 1000),
        "text": seg.text.strip(),
    })

if not sentences:
    print("ERROR: 没转出任何文字，视频里可能没有人声")
    sys.exit(1)

# ---------- 5. 落盘（文件名和旧版一样，后面步骤不用改） ----------
out_path = os.path.join(BASE, "sentences.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sentences, f, ensure_ascii=False, indent=2)

print("识别到 %d 个句子片段" % len(sentences))
print("已写入:", out_path)
print("\n前 10 句预览：")
for s in sentences[:10]:
    sec = s["start"] / 1000
    print("  [%02d:%02d] %s" % (sec // 60, sec % 60, s["text"][:50]))
