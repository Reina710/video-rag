# -*- coding: utf-8 -*-
"""
第二步（多模态版）：把 ASR 语音文字 + OCR 画面文字 一起存进 Qdrant。

和旧版 qdrant_store.py 的区别：
  旧版：只存 sentences.json（ASR 单条线）
  新版：ASR（sentences.json）+ OCR（frames_ocr.json）双分支，
        每条 point 的 payload 里加 source 字段（"asr" / "ocr"）

这是复现原项目 VideoSegment{transcript + ocrTexts} 的关键——
两条线各存各的，检索时都能命中，问答时按 source 区分证据来源。

关键设计点：
  - 为什么 ASR 和 OCR 分开存而不是合并成一段：语音和画面的时间粒度不同
    （ASR 按句、OCR 按关键帧），分开存检索更准，问答时再按时间戳对齐
  - source 字段是"证据溯源"的基础：回答里能说清"这条证据来自语音还是画面"
"""
import json
import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------- 1. 读 key ----------
import settings

key = settings.get("DASHSCOPE_KEY")
if not key:
    raise SystemExit("没找到 DASHSCOPE_KEY（写在 .env 或环境变量里）")

embed_client = OpenAI(
    api_key=key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def embed(text):
    resp = embed_client.embeddings.create(model="text-embedding-v3", input=text)
    return resp.data[0].embedding

# ---------- 2. 连 Qdrant ----------
client = QdrantClient(**settings.qdrant_kwargs())

COLLECTION = "video_sentences"
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
print("collection 已重建:", COLLECTION)

# ---------- 3. 读两份数据，合并 ----------
base = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base, "sentences.json"), encoding="utf-8") as f:
    asr = json.load(f)
with open(os.path.join(base, "frames_ocr.json"), encoding="utf-8") as f:
    ocr = json.load(f)

print("ASR 句子 %d 条，OCR 帧 %d 条" % (len(asr), len(ocr)))

# 统一格式成 [(start毫秒, end毫秒, text, source)]
items = []
for s in asr:
    items.append((s["start"], s["end"], s["text"], "asr"))
for o in ocr:
    # OCR 帧没有 end，用 start + 30 秒（抽帧间隔）估算
    items.append((o["start"], o["start"] + 30000, o["text"], "ocr"))

# ---------- 4. 逐条转向量入库 ----------
points = []
for i, (start, end, text, source) in enumerate(items):
    vec = embed(text)
    points.append(PointStruct(
        id=i,
        vector=vec,
        payload={"start": start, "end": end, "text": text, "source": source},
    ))

client.upsert(collection_name=COLLECTION, points=points)
print("已存入 %d 个向量（ASR %d + OCR %d）" % (len(points), len(asr), len(ocr)))

# ---------- 5. 验证：查一个问题，看 ASR/OCR 都能命中吗 ----------
query_vec = embed("无职转生3 是什么题材")
hits = client.query_points(
    collection_name=COLLECTION, query=query_vec, limit=5,
).points

print("\n查询「无职转生3 是什么题材」的 top 5 命中：")
for h in hits:
    p = h.payload
    sec = p["start"] / 1000
    src = p.get("source", "?")
    print("  [%02d:%02d] (%s) %.3f %s" % (sec // 60, sec % 60, src, h.score, p["text"][:40]))
