# -*- coding: utf-8 -*-
"""
第一步：视频 -> 带时间戳的文字片段。

现在先用【官方公开视频链接】把整条链路跑通，让你看到真实结果长啥样。
等你以后部署到云服务器，再把你自己的本地视频传进 MinIO 换成你自己的链接。

面试可讲的点（[面试] 标注）：
  - 为什么不用本地 Whisper：45 分钟视频 40 秒 vs 15 分钟，20 倍
  - 为什么结果要带时间戳：后面问答"证据绑定"靠它
"""
import os
import sys
import json
from urllib.request import urlopen

import settings

# ---------- 1. 读 key ----------
key = settings.get("DASHSCOPE_KEY")
if not key:
    print("ERROR: 没找到 DASHSCOPE_KEY（写在 .env 或环境变量里）")
    sys.exit(1)

import dashscope
dashscope.api_key = key
from dashscope.audio.asr import Transcription

# ---------- 2. 视频地址（先换成官方公开的，公网能下载） ----------
# 阿里官方公开样例（带人声的 wav，能出转写结果）
video = "https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female.wav"

# ---------- 3. 调 Paraformer 转写 ----------
task = Transcription.async_call(
    model="paraformer-v2",
    file_urls=[video],
    language_hints=["zh"],
)
result = Transcription.wait(task)

# ---------- 4. 看任务状态 ----------
results = result["output"].get("results", [])
if not results:
    print("无结果，完整返回:", json.dumps(result, ensure_ascii=False)[:1500])
    sys.exit(1)

first = results[0]
if first.get("subtask_status") != "SUCCEEDED":
    print("子任务失败:", first.get("code"), first.get("message"))
    sys.exit(1)

# ---------- 5. 下载 transcription_url，里面才是带时间戳的句子 ----------
trans_url = first["transcription_url"]
trans = json.loads(urlopen(trans_url, timeout=30).read().decode("utf-8"))

# 结构：transcripts[].sentences[]，begin_time/end_time 是毫秒
sentences = []
for t in trans.get("transcripts", []):
    for s in t.get("sentences", []):
        sentences.append({
            "start": s["begin_time"],
            "end": s["end_time"],
            "text": s["text"],
        })

# ---------- 6. 落盘 ----------
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentences.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sentences, f, ensure_ascii=False, indent=2)

print("识别到 %d 个句子片段" % len(sentences))
print("已写入:", out_path)
for s in sentences[:10]:
    sec = s["start"] / 1000
    print("  [%02d:%02d] %s" % (sec // 60, sec % 60, s["text"][:40]))
