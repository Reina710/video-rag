# -*- coding: utf-8 -*-
"""
第三步：问答。用户提问 -> 向量检索 -> 喂给 DeepSeek -> 回答带时间戳。

这是整个项目的最后一环，把前两步串起来：
  video_to_text.py（视频->文字） + qdrant_store.py（文字->向量）
  -> 本文件（问题->检索->回答）

面试可讲的点（[面试] 标注）：
  - 时间戳不是模型编的，是 Qdrant 检索出来的 payload 自带的（证据绑定）
  - 检索到"最像的几句话"喂给模型，模型只负责组织语言，不负责"找"
"""
import json
from openai import OpenAI
from qdrant_client import QdrantClient

# ---------- 1. 读 key ----------
import settings

dashscope_key = settings.get("DASHSCOPE_KEY")
deepseek_key = settings.get("DEEPSEEK_KEY")

# ---------- 2. 两个客户端 ----------
# embedding 客户端：把问题转成向量（和存句子时用同一个模型，才能比）
embed_client = OpenAI(
    api_key=dashscope_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 对话客户端：DeepSeek，负责组织语言回答
chat_client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")

# Qdrant 客户端
qdrant = QdrantClient(host="127.0.0.1", port=6333)

COLLECTION = "video_sentences"


def search(question, limit=3):
    """问题 -> 向量 -> Qdrant 检索 top N 个最像的句子。"""
    # [面试] 问题必须用"同一个 embedding 模型"转，才能和已存的句子比较
    qvec = embed_client.embeddings.create(
        model="text-embedding-v3", input=question
    ).data[0].embedding

    hits = qdrant.query_points(
        collection_name=COLLECTION, query=qvec, limit=limit
    ).points

    # 抽出来，整理成"带时间戳的原文"喂给模型
    context = []
    for h in hits:
        p = h.payload
        sec = p["start"] / 1000
        context.append("[第 %02d:%02d 秒] %s" % (sec // 60, sec % 60, p["text"]))
    return context, hits


def answer(question):
    """完整回答：检索 + 生成。"""
    context, hits = search(question)

    # [面试] 这是 RAG 的核心：把检索到的"证据"塞进提示词，模型照着答
    system = (
        "你是视频内容问答助手。下面是从视频里检索到的、和用户问题最相关的片段，"
        "每个片段都带时间戳。请只用这些片段回答，并指出答案来自视频的哪个时间点。"
        "如果片段里没有答案，就直说'视频里没有提到'，不要编造。\n\n"
        + "\n".join(context)
    )

    resp = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


# ---------- 主程序：连问两个问题演示 ----------
if __name__ == "__main__":
    for q in ["语音实验室说了什么", "视频里提到了哪家公司"]:
        print("问题：%s" % q)
        print("回答：%s" % answer(q))
        print("-" * 50)
