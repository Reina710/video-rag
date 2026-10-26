# -*- coding: utf-8 -*-
"""
分段摘要：把长视频按 5 分钟切块，每块生成一段摘要，拼成「全片目录」。

为什么要它（长视频 RAG 的核心问题：时间坍缩）：
    向量检索最容易召回的是开场总述和关键词密集处，结果证据全挤在视频前几分钟，
    后面十几分钟根本没进上下文。我试过"时间覆盖约束"，治标——模型看到的还是零散句子，
    它不知道自己漏了什么。

    分段摘要是治本：先把全片压成一份「每段讲了什么」的目录，
    模型先看目录知道自己漏了什么，再决定深入哪一段。

落地方式：
    摘要 + 关键词存 SQLite，按 video_id 缓存，同一个视频只生成一次；
    检索时先命中摘要段，再用该段的时间范围回捞原文（small-to-big）。

用法：
    from chunks import ensure
    chunks = ensure(video_id, media_id)   # 有缓存直接返回，没有才现建
"""
import json
import os
import re

import settings
import storage

WINDOW_SEC = 300          # 5 分钟一段
MAX_ASR_CHARS = 3000      # 单段喂给模型的语音文本上限，防超长
MAX_OCR_CHARS = 1200      # 画面文字上限
MODEL = "deepseek-chat"


def _env_value(key):
    """读配置：环境变量优先，其次项目内 .env。"""
    return settings.get(key)


def _client():
    from openai import OpenAI
    return OpenAI(
        api_key=_env_value("DEEPSEEK_KEY"),
        base_url="https://api.deepseek.com",
        timeout=60,
        max_retries=2,
    )


def _docs(media_id=None):
    """复用 retrieval 的语料缓存，避免再拉一次 Qdrant。"""
    from retrieval import _corpus
    docs = _corpus()["docs"]
    if media_id is not None:
        docs = [d for d in docs if d["media_id"] == int(media_id)]
    return docs


def _mmss(ms):
    sec = int(ms // 1000)
    return "%02d:%02d" % (sec // 60, sec % 60)


def _summarize(lo, hi, asr_text, ocr_text):
    """调一次 LLM 生成这一段摘要。失败返回空串，不阻断。"""
    prompt = (
        "下面是视频第 %s 到 %s 分钟的语音转写和画面文字。\n"
        "请用 JSON 输出：\n"
        '{"summary": "2-3 句话概括这一段讲了什么（涉及哪些作品/话题/观点）", '
        '"keywords": ["出现的关键作品或主题"]}\n'
        "要求：\n"
        "1) 只写材料里出现过的内容，不要编造；\n"
        "2) summary 控制在 80 字以内，写给想快速定位「这段讲了什么」的人；\n"
        "3) keywords 最多 8 个；\n"
        "4) 只输出 JSON，不要解释。\n\n"
        "【语音转写】\n%s\n\n【画面文字】\n%s"
        % (_mmss(lo), _mmss(hi), asr_text or "（无）", ocr_text or "（无）")
    )
    try:
        resp = _client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        data = _parse_json(resp.choices[0].message.content or "")
        return (data.get("summary") or "").strip(), [str(k) for k in (data.get("keywords") or [])][:8]
    except Exception:
        return "", []


def _parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


def build(video_id, media_id=None, window_sec=WINDOW_SEC, force=False, on_progress=None):
    """为某个视频生成分段摘要并落库，返回 [{startMs, endMs, summary, keywords}]。"""
    if not force:
        cached = storage.load_chunks(video_id)
        if cached:
            return cached

    docs = _docs(media_id)
    if not docs:
        return []

    total = max(d["end"] for d in docs)
    step = window_sec * 1000
    count = max(1, (total + step - 1) // step)

    out = []
    for i in range(count):
        lo, hi = i * step, min(total, (i + 1) * step)
        seg = [d for d in docs if lo <= d["start"] < hi]
        if not seg:
            out.append({"startMs": lo, "endMs": hi,
                        "summary": "（这段没有识别到语音或画面文字）", "keywords": []})
        else:
            asr_text = " ".join(d["text"] for d in seg if d["source"] == "asr")[:MAX_ASR_CHARS]
            ocr_text = " | ".join(d["text"].replace("\n", " ")
                                  for d in seg if d["source"] == "ocr")[:MAX_OCR_CHARS]
            summary, kws = _summarize(lo, hi, asr_text, ocr_text)
            out.append({"startMs": lo, "endMs": hi,
                        "summary": summary or "（摘要生成失败）", "keywords": kws})
        if on_progress:
            on_progress(i + 1, count)

    storage.save_chunks(video_id, out)
    return out


def ensure(video_id, media_id=None):
    """有缓存用缓存，没有就现建。"""
    cached = storage.load_chunks(video_id)
    if cached:
        return cached
    return build(video_id, media_id=media_id)


def as_text(chunks):
    """把分段摘要拼成给模型看的目录。"""
    if not chunks:
        return ""
    lines = []
    for c in chunks:
        kws = "、".join(c.get("keywords") or [])
        line = "[%s-%s] %s" % (_mmss(c["startMs"]), _mmss(c["endMs"]), c.get("summary") or "")
        if kws:
            line += "（关键词：%s）" % kws
        lines.append(line)
    return "\n".join(lines)
