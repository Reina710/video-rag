# -*- coding: utf-8 -*-
"""
检索层：双路召回（向量 + BM25）→ RRF 合并 → rerank 精排。

和校园问答项目的关系（同一套方法论的第二次落地）：
    校园项目在干净文本块上验证了「单路 → 融合 → 精排」的收益；
    视频项目把这套方法论搬到"脏数据 + 时间维度"：
      - 脏数据：ASR 有错别字和繁体输出、OCR 是画面断句，纯向量对专有名词不稳
      - 时间维度：每条证据带 start/end，命中之后还要能跳回视频那一秒
    所以这里复用同一套三段式，只是召回单元从"知识块"换成"带时间戳的片段"。
    迁移过程中真正暴露的问题是：方法能搬，前提条件不能想当然（见 _simplify 与 b 参数）。

三段式各自解决什么（和校园项目一致）：
    vector  双塔（bi-encoder）：便宜、求全，负责"从 455 条里捞出可能相关的"
    bm25    字面匹配：对专有名词（番剧名）敏感，补向量的短板
    rrf     融合两路排名，不用归一化分数（不同检索器的分数量纲不一样，没法加权）
    rerank  交叉编码：问题和片段拼在一起读，负责"这 10 条里到底哪 3 条最该给模型"

工程约定：
    1. rerank 失败返回 None，绝不抛异常 —— 上层降级回 RRF 顺序
    2. vector 挂了（Qdrant/embedding 不可用）→ 只用 BM25，链路不断
    3. 分数只在同一 query 内可比，不拿来当阈值
"""
import math
import os
import threading
from collections import Counter

import settings

COLLECTION = "video_sentences"

RECALL_K = 20      # 每路宽召回条数（后续给 RRF 合并）
RERANK_N = 10      # 送进 rerank 的候选数
RERANK_MODEL = "gte-rerank-v2"

# 上下文扩展窗口：ASR 一句只有 2~3 秒，关键结论和它的宾语常被切在两句话里
# （「本季最强的」/「碧蓝之海」），只给命中的那一句，模型就没法回答。
# 往后多看 20 秒：结论后面才是具体作品名/解释；往前看 8 秒补上文。
WINDOW_BEFORE_MS = 8000
WINDOW_AFTER_MS = 20000
# 窗口总长封顶：再长就只是噪声，而且时间跨度一大，锚点就失去意义
MAX_WINDOW_MS = 40000
# 只有锚点离得这么近的窗口才合并；差 30 秒的两帧根本不是一个位置
MERGE_ANCHOR_GAP_MS = 5000


# ---------- 0. 清代理（Windows 系统代理会拦 127.0.0.1） ----------
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
_no = os.environ.get("NO_PROXY", "")
if "127.0.0.1" not in _no:
    os.environ["NO_PROXY"] = ",".join(filter(None, [_no, "127.0.0.1", "localhost"]))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


def _env_value(key):
    """读配置：环境变量优先，其次项目内 .env。不依赖任何其他项目的路径。"""
    return settings.get(key)


_CLIENTS = {}


def _qdrant():
    if "q" not in _CLIENTS:
        from qdrant_client import QdrantClient
        _CLIENTS["q"] = QdrantClient(host="127.0.0.1", port=6333)
    return _CLIENTS["q"]


def _embed():
    if "e" not in _CLIENTS:
        from openai import OpenAI
        _CLIENTS["e"] = OpenAI(
            api_key=_env_value("DASHSCOPE_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _CLIENTS["e"]


# ---------- 1. 分词 ----------
_STOP = set("""
的 了 吗 呢 啊 吧 呀 哦 哈 嗯 个 是在 和 与 就 都 也 很 还 又 但 而 所以 因为 如果 这个 那个
这 那 你 我 他 她 它 我们 你们 他们 什么 怎么 哪 哪些 有 没有 一个 一种 一些 可以 就是 不是
""".split())

# 繁简转换器（懒加载）。faster-whisper 的中文输出默认是繁体，而用户提问是简体：
# 「漫畫」和「漫画」在 BM25 里是两个完全不同的词，字面召回直接归零，
# 而 embedding 对繁简不敏感 —— 结果就是「向量能召回、BM25 召不回」，
# RRF 融合时往池子里注入的全是噪声，Top-1 反而比纯向量更低。
# 实测：8 题黄金集，BM25 对正确区间的召回 7/8 → 8/8，融合 hit@1 75% → 87.5%。
_CC = None
_CC_TRIED = False


def _simplify(text):
    """繁 → 简。装不上 opencc 就原样返回，链路不因缺依赖而断。"""
    global _CC, _CC_TRIED
    if not _CC_TRIED:
        _CC_TRIED = True
        try:
            from opencc import OpenCC

            _CC = OpenCC("t2s")
        except Exception:
            _CC = None
    return _CC.convert(text) if _CC is not None else text


def _tokenize(text):
    """繁转简 + jieba 分词 + 去停用词 + 去纯标点。中文关键词靠这一步才对得上。"""
    import re

    import jieba

    text = _simplify((text or "").lower())
    out = []
    for w in jieba.lcut(text):
        w = w.strip()
        if not w or w in _STOP:
            continue
        if re.fullmatch(r"[\W_]+", w):
            continue
        out.append(w)
    return out


# ---------- 2. 语料缓存（455 条，一次拉全，tokenize 只做一次） ----------
_CORPUS = None
_CORPUS_LOCK = threading.Lock()


def _corpus():
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    with _CORPUS_LOCK:
        if _CORPUS is not None:
            return _CORPUS
        client = _qdrant()
        docs = []
        offset = None
        while True:
            pts, offset = client.scroll(
                COLLECTION, limit=500, offset=offset, with_payload=True
            )
            for p in pts:
                pl = p.payload or {}
                text = pl.get("text", "")
                docs.append({
                    "key": "%s|%s|%s" % (pl.get("start"), pl.get("source"), text[:40]),
                    "start": pl.get("start", 0),
                    "end": pl.get("end", 0),
                    "source": (pl.get("source") or "asr").lower(),
                    "text": text,
                    "media_id": pl.get("media_id"),
                    "tokens": _tokenize(text),
                })
            if offset is None:
                break
        _CORPUS = {"docs": docs}
        return _CORPUS


def refresh_corpus():
    """上传/删除视频后调用，让 BM25 语料跟着更新。"""
    global _CORPUS
    with _CORPUS_LOCK:
        _CORPUS = None


# ---------- 3. 单路检索 ----------
def _hit(doc, **extra):
    return {
        "key": doc["key"],
        "start": doc["start"],
        "end": doc["end"],
        "source": doc["source"],
        "text": doc["text"],
        "media_id": doc.get("media_id"),
        **extra,
    }


def vector_search(query, limit=RECALL_K, media_id=None, source=None, time_range=None):
    """向量检索（Qdrant）。失败抛异常，由上层降级。

    time_range=(start_ms, end_ms) 可选：只在某个时间桶里检索，
    用于给长视频补"后段证据"，防止结果全挤在开头（时间坍缩）。
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

    qvec = _embed().embeddings.create(
        model="text-embedding-v3", input=query
    ).data[0].embedding

    conds = []
    if media_id is not None:
        conds.append(FieldCondition(key="media_id", match=MatchValue(value=int(media_id))))
    if source:
        conds.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if time_range:
        conds.append(FieldCondition(
            key="start", range=Range(gte=int(time_range[0]), lt=int(time_range[1]))
        ))
    qfilter = Filter(must=conds) if conds else None

    hits = _qdrant().query_points(
        collection_name=COLLECTION, query=qvec, query_filter=qfilter, limit=limit
    ).points
    return [{
        "key": "%s|%s|%s" % (h.payload.get("start"), h.payload.get("source"), h.payload.get("text", "")[:40]),
        "start": h.payload.get("start", 0),
        "end": h.payload.get("end", 0),
        "source": (h.payload.get("source") or "asr").lower(),
        "text": h.payload.get("text", ""),
        "media_id": h.payload.get("media_id"),
        "vec": float(h.score),
    } for h in hits]


def keyword_search(query, limit=RECALL_K, media_id=None, source=None):
    """BM25 关键词检索（k1=1.5, b=0.4）。

    为什么中文要用 jieba：不分词的话，字符级匹配会把「无职转生」和「无职」当成两回事，
    IDF 也算不对。

    为什么 b 不能照搬校园项目的 0.75：
        长度归一 b 的假设是「文档越长，命中的词越可能是巧合，要打折」。
        这条假设在知识块上成立，在 2~3 秒的 ASR 行上不成立 —— 这里每条只有
        3~4 个词，长度差异来自**语速**，不来自信息量。b=0.75 会把
        「我最愛的漫畫」这种 7 个字的短句权重放大 5 倍，压过真正有上下文的片段。
        实测（8 题）：b=0.75 融合 hit@1 87.5%，b=0.4 时 100%。
    """
    docs = _corpus()["docs"]
    if media_id is not None:
        docs = [d for d in docs if d["media_id"] == int(media_id)]
    if source:
        docs = [d for d in docs if d["source"] == source]
    if not docs:
        return []

    q_terms = _tokenize(query)
    if not q_terms:
        return []

    n = len(docs)
    avgdl = sum(len(d["tokens"]) for d in docs) / max(1, n)
    df = Counter()
    for d in docs:
        for t in set(d["tokens"]):
            df[t] += 1

    k1, b = 1.5, 0.4
    scored = []
    for d in docs:
        tf = Counter(d["tokens"])
        dl = len(d["tokens"]) or 1
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        if s > 0:
            scored.append((s, d))
    scored.sort(key=lambda x: -x[0])
    return [_hit(d, bm25=round(s, 4)) for s, d in scored[:limit]]


# ---------- 4. RRF 融合 ----------
def rrf_fuse(ranked_lists, k=60):
    """Reciprocal Rank Fusion：只用排名，不用分数。

    为什么不用加权分数：向量余弦是 0~1，BM25 是无上界，量纲不同，加权等于拍脑袋。
    RRF 只看"排第几"，天然可比，也不怕某个检索器分数尺度变化。

    顺带一个反证：两路都修好之后，手工给向量 2:1 加权反而掉点（8 题 hit@1 100% → 87.5%）。
    融合权重不是调出来的，是两路各自可信之后自然就对的。
    """
    score = {}
    doc_of = {}
    for lst in ranked_lists:
        for rank, d in enumerate(lst, 1):
            key = d["key"]
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank)
            doc_of.setdefault(key, d)
    merged = [dict(doc_of[key], rrf=round(score[key], 6)) for key in score]
    merged.sort(key=lambda x: -x["rrf"])
    return merged


# ---------- 5. rerank 精排 ----------
def rerank(query, candidates, top_n=None):
    """交叉编码精排。失败返回 None（上层降级回 RRF 顺序）。"""
    if not candidates:
        return candidates
    try:
        from dashscope import TextReRank
    except ImportError:
        return None

    docs = [
        "[%s] %s" % ("语音" if c["source"] == "asr" else "画面", c["text"])
        for c in candidates
    ]
    try:
        resp = TextReRank.call(
            model=RERANK_MODEL,
            query=query,
            documents=docs,
            top_n=len(docs),
            return_documents=False,
            api_key=_env_value("DASHSCOPE_KEY"),
        )
    except Exception:
        return None

    if getattr(resp, "status_code", None) != 200:
        return None
    try:
        score_of = {r["index"]: r["relevance_score"] for r in resp.output["results"]}
    except (KeyError, TypeError):
        return None

    out = [dict(c, rerank=score_of.get(i, 0.0)) for i, c in enumerate(candidates)]
    out.sort(key=lambda x: -x["rerank"])
    return out[:top_n] if top_n else out


# ---------- 6. 对外入口 ----------
def search_raw(query, limit=5, media_id=None, mode="hybrid", source=None):
    """检索但不扩窗 —— 评测用。

    为什么评测不用扩窗版：窗口有 28 秒，几乎什么都能"重叠"上黄金区间，
    指标会全部饱和到 100%，失去区分度。扩窗是检索之后的**后处理**，
    它的收益用"回答有没有补全宾语"来验证，不拿 hit@K 量。
    """
    vec = vector_search(query, max(limit, RECALL_K), media_id, source)
    if mode == "vector":
        return vec[:limit]

    kw = keyword_search(query, max(limit, RECALL_K), media_id, source)
    fused = rrf_fuse([vec, kw])

    if mode == "hybrid+rerank" and fused:
        rr = rerank(query, fused[:RERANK_N])
        if rr is not None:
            fused = rr
    return fused[:limit]


def search(query, limit=5, media_id=None, mode="hybrid", source=None):
    """生产用：检索 → 扩上下文窗口 → rerank。

    顺序很重要（踩过的坑）：
        如果先 rerank 再扩窗，「本季最强的」这种短句会因为"它没回答问题"
        被 rerank 排掉 —— 可它恰恰是答案的前半句，宾语在下一句。
        所以必须先把候选扩成窗口，让 rerank 看到完整上下文，再定序。

    mode: vector | hybrid | hybrid+rerank
    """
    vec = vector_search(query, max(limit, RECALL_K), media_id, source)
    if mode == "vector":
        return expand_windows(vec[:RERANK_N], media_id=media_id)[:limit]

    kw = keyword_search(query, max(limit, RECALL_K), media_id, source)
    fused = rrf_fuse([vec, kw])

    # 先扩窗（答案可能在相邻句子里），再让 rerank 看完整上下文定序
    cands = expand_windows(fused[:RERANK_N], media_id=media_id)
    if mode == "hybrid+rerank" and cands:
        rr = rerank(query, cands)
        if rr is not None:
            cands = rr
    return cands[:limit]


def expand_windows(hits, media_id=None, before_ms=WINDOW_BEFORE_MS, after_ms=WINDOW_AFTER_MS):
    """把命中的片段扩成上下文窗口，但**锚点保持不变**。

    为什么必须扩（视频 RAG 和文本 RAG 最大的差别之一）：
        ASR 一行只有 2~3 秒，「本季最强的」和它的答案「碧蓝之海」是两行。
        检索命中的往往正好是"结论"那半句，宾语在下一行。

    为什么锚点要单独存（踩过的坑）：
        最开始窗口 start 直接用"第一个相邻句的起点"，而 OCR 一帧覆盖 30 秒、
        帧和帧首尾相接，窗口就**链式合并**成 30 秒~170 秒的一大坨：
        时间戳显示 00:30，内容其实在 01:30 —— 点过去完全对不上。
        现在 start/end 只描述范围，anchor 永远等于命中句起点，前端和证据都用 anchor。

    两个防"吞太多"的约束：
        1. 只收「起点落在窗口内」的片段（不是"有重叠就收"），避免边界帧被拉进来
        2. 窗口总长封顶 MAX_WINDOW_MS
    """
    if not hits:
        return []

    groups = {}
    for d in _corpus()["docs"]:
        groups.setdefault((d["media_id"], d["source"]), []).append(d)
    for g in groups.values():
        g.sort(key=lambda d: d["start"])

    windows = []
    for idx, h in enumerate(hits):
        mid = h.get("media_id") if h.get("media_id") is not None else media_id
        group = groups.get((mid, h["source"])) or []
        if h["source"] == "ocr":
            # OCR 一帧本身就覆盖 30 秒，再扩只会把相邻帧拉进来：
            # 相邻帧画面已经跳了，内容不连续，不算上下文，还会制造重复条目。
            lo, hi = h["start"], h["start"] + 1
        else:
            lo = max(0, h["start"] - before_ms)
            hi = min(h["start"] + MAX_WINDOW_MS, h["end"] + after_ms)
        # 关键：只收"起点落在窗口内"的片段。OCR 一帧覆盖 30 秒，
        # 用"有重叠就收"会把 90 秒那帧的窗口一路拉到 60 秒、30 秒。
        picked = [d for d in group if lo <= d["start"] < hi]
        if not picked:
            picked = [{"start": h["start"], "end": h["end"], "text": h["text"]}]
        windows.append({
            "start": min(d["start"] for d in picked),
            "end": max(d["end"] for d in picked),
            "anchor": h["start"],          # ← 点击跳转 / 证据时间戳都用这个
            "source": h["source"],
            "text": " ".join(d["text"] for d in picked),
            "media_id": mid,
            # 记住这条窗口来自第几名命中：合并后要按这个排回相关性顺序，
            # 否则扩窗会把"最相关"变成"时间最早"，排序信息就丢了
            "rank": idx,
        })

    # 只有"锚点很近"的窗口才合并：锚点差 30 秒的两帧根本不是一个位置，
    # 合并了时间戳就只能二选一，必然有一个对不上。
    windows.sort(key=lambda w: (w["media_id"] if w["media_id"] is not None else -1,
                                w["source"], w["anchor"]))
    merged = []
    for w in windows:
        if merged:
            last = merged[-1]
            if (last["media_id"] == w["media_id"] and last["source"] == w["source"]
                    and abs(w["anchor"] - last["anchor"]) <= MERGE_ANCHOR_GAP_MS):
                last["end"] = max(last["end"], w["end"])
                last["start"] = min(last["start"], w["start"])
                last["anchor"] = min(last["anchor"], w["anchor"])
                last["rank"] = min(last["rank"], w["rank"])
                if w["text"] not in last["text"]:
                    last["text"] += " " + w["text"]
                continue
        merged.append(dict(w))

    # 按相关性排名还原顺序（关键：否则 limit 会取到时间最早的，而不是最相关的）
    merged.sort(key=lambda w: (w["rank"], w["anchor"]))
    return merged


COVERAGE_BUCKETS = 4


def _video_span(media_id=None):
    docs = _corpus()["docs"]
    if media_id is not None:
        docs = [d for d in docs if d["media_id"] == int(media_id)]
    return max((d["end"] for d in docs), default=0)


def add_time_coverage(hits, query, media_id=None, buckets=COVERAGE_BUCKETS):
    """保证证据在时间轴上铺开，别全挤在开头。

    长视频 RAG 的典型毛病叫「时间坍缩」：向量最相似的片段往往是开场介绍/总述，
    于是模型只看到前几分钟，后面十几分钟的内容根本没进上下文。
    这里把视频切成 N 个时间桶，哪个桶一条证据都没有，就在那个桶里补一条最相关的。
    """
    total = _video_span(media_id)
    if not total or total < buckets * 1000:
        return hits

    size = total / buckets
    covered = set()
    for h in hits:
        covered.add(min(buckets - 1, int(h.get("anchor", h["start"]) // size)))

    extra = []
    for b in range(buckets):
        if b in covered:
            continue
        lo, hi = int(b * size), int((b + 1) * size)
        try:
            cand = vector_search(query, 1, media_id=media_id, time_range=(lo, hi))
        except Exception:
            cand = []
        if cand:
            extra.extend(expand_windows(cand[:1], media_id=media_id))
    return hits + extra


def retrieve_hybrid(query, media_id=None, asr_final=8, ocr_final=4, mode="hybrid+rerank"):
    """Agent 用：ASR / OCR 各自跑一遍完整流水线（含扩窗、精排），再补全时间覆盖。

    为什么还分开：OCR 只有 30+ 条、ASR 有 400+ 条，合成一个池子按相似度捞，
    OCR 会被淹没。分开各自 hybrid + 扩窗 + rerank，保证画面文字也能进证据池。
    """
    out = []
    for src, final in (("asr", asr_final), ("ocr", ocr_final)):
        out.extend(search(query, final, media_id=media_id, mode=mode, source=src))
    # 时间覆盖：哪个时间段一条证据都没有，就补一条，避免只看开头
    out = add_time_coverage(out, query, media_id=media_id)
    return [{
        "startMs": h["start"],      # 窗口范围（给模型看上下文边界）
        "endMs": h["end"],
        "anchorMs": h.get("anchor", h["start"]),  # ← 证据时间戳 / 点击跳转用它
        "source": h["source"].upper(),
        "text": h["text"],
    } for h in out]
