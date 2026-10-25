# -*- coding: utf-8 -*-
"""
检索层评测：只评「检索有没有捞到正确的证据」，不评生成。

为什么要单独评检索：
  端到端指标（evaluate.py）只看最终答案好不好，定位不到问题出在
  「检索没捞到」还是「生成没写好」。这一层用人工标定好的【黄金时间戳】
  当标准答案，把检索单独拎出来量：

    hit@K      Top-K 里有没有落在正确时间区间的证据（有/无）
    recall@K   多个正确区间被覆盖的比例（一个问题可能有几处证据）
    MRR        第一条命中证据排在第几名的倒数（越接近 1 越好）

  更重要的是：同一套题可以对比不同检索配置，用数据决定要不要上融合/精排，
  而不是"别人都有所以我也加"。

用法：
    python eval_retrieval.py                          # 默认 hybrid+rerank
    python eval_retrieval.py --compare                # 三种配置 × hit@1/3/5 + 耗时
    python eval_retrieval.py --content                # 扩窗前后，答案有没有真的进证据
    python eval_retrieval.py --mode vector --top-k 5
    python eval_retrieval.py --media-id 5 --verbose
"""
import argparse
import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from retrieval import _simplify as simplify_text
from retrieval import search as retrieval_search_windowed
from retrieval import search_raw as retrieval_search

# ---------- 黄金题集：问题是人工写的，时间区间是人工看完视频标的 ----------
# 区间单位毫秒；一个问题可以标多个区间（同一答案在不同位置出现）
GOLDEN = [
    {
        "name": "本季高分作数量",
        "goal": "本季有几部高分作？",
        "ranges": [[78000, 100000]],
    },
    {
        "name": "三部作品清单",
        "goal": "视频里提到的三部作品分别是哪三部？",
        "ranges": [[78000, 105000]],
    },
    {
        "name": "穹庐下的魔女题材",
        "goal": "《穹庐下的魔女》是什么题材？",
        "ranges": [[85000, 105000]],
    },
    {
        "name": "弹丸论破发布时间",
        "goal": "《弹丸论破2x2》什么时候发布？",
        "ranges": [[290000, 320000]],
    },
    {
        "name": "猫与龙类型",
        "goal": "《猫与龙》是什么类型的作品？",
        "ranges": [[345000, 370000]],
    },
    {
        "name": "猫与龙出现位置",
        "goal": "视频里哪些地方提到了《猫与龙》？",
        # 两处：05:57 语音介绍 + 14:00 画面文字，考 recall 多区间
        "ranges": [[345000, 370000], [835000, 850000]],
    },
    {
        "name": "本季最强",
        "goal": "视频说本季最强的是哪部作品？",
        "ranges": [[390000, 410000]],
    },
    {
        "name": "最爱的漫画",
        "goal": "作者最喜欢的漫画是哪部？",
        "ranges": [[905000, 925000]],
    },
    # ---- 下面 8 题是 2026-10 补充的：原集只有 8 题，hit@K 一到 top5 就饱和 ----
    {
        "name": "开头三件事",
        "goal": "视频开头说，看完这期视频一定要记得的三件事是什么？",
        "ranges": [[62000, 74000]],
    },
    {
        "name": "高分作评分段",
        "goal": "本季那几部高分作的评分大概在什么段位？",
        "ranges": [[78000, 90000]],
    },
    {
        "name": "画完这个再去死的四字评价",
        "goal": "视频用哪四个字评价《画完这个再去死》？",
        "ranges": [[160000, 172000]],
    },
    {
        "name": "黑马成精品的劲头",
        "goal": "本季黑马作品是靠着什么劲头成为值得一看的精品的？",
        "ranges": [[198000, 208000]],
    },
    {
        "name": "猫与龙的神父",
        "goal": "《猫与龙》世界观里的孤儿院神父是个什么样的人？",
        "ranges": [[362000, 375000]],
    },
    {
        "name": "二十世纪电器目录的定位",
        "goal": "《二十世纪电器目录》被形容成一部什么样的作品？",
        "ranges": [[496000, 528000]],
    },
    {
        "name": "再见啦啦的核心问题",
        "goal": "视频认为《再见啦啦》最核心的问题是什么？",
        "ranges": [[778000, 798000]],
    },
    {
        "name": "向日葵马戏团的三条线",
        "goal": "《向日葵马戏团》被夸的三条线分别是什么？",
        "ranges": [[448000, 475000]],
    },
    # ---- 两题只靠画面文字（ASR 完全没提），用来验证 OCR 这一路是否真的起作用 ----
    {
        "name": "复联5谁回归",
        "goal": "视频画面里说，哪个角色要在《复仇者联盟5：毁灭之日》里回归？",
        "ranges": [[330000, 360000]],
    },
    {
        "name": "马戏盛典冠军",
        "goal": "第24届马戏盛典日本大赛的冠军是谁？",
        "ranges": [[896000, 930000]],
    },
]

MODES = [
    ("vector", "① 纯向量"),
    ("hybrid", "② 向量 + BM25 + RRF"),
    ("hybrid+rerank", "③ 再加 rerank"),
]

# 答案关键词：用于「内容级」评测——命中片段/窗口里到底有没有答案。
# hit@K 只回答"名次对不对"，不回答"拿到的那块里有没有内容"：
# 一句「本季最强的」排在第一名，名次满分，但它本身没有答案（宾语在下一句）。
# 一条答案有多个要点就写多个词，按覆盖率算。
ANSWER_KEYS = {
    "本季高分作数量": ["三部"],
    "三部作品清单": ["穹庐", "无职", "画完"],
    "穹庐下的魔女题材": ["历史"],
    "弹丸论破发布时间": ["2026"],
    "猫与龙类型": ["童话", "群像"],
    "本季最强": ["碧蓝之海"],
    "最爱的漫画": ["漫画"],
    "开头三件事": ["商量", "写信", "怀疑"],
    "高分作评分段": ["8.0"],
    "画完这个再去死的四字评价": ["真情流露"],
    "黑马成精品的劲头": ["热爱"],
    "猫与龙的神父": ["好人"],
    "二十世纪电器目录的定位": ["时代"],
    "再见啦啦的核心问题": ["空洞"],
    "向日葵马戏团的三条线": ["团魂", "成长", "竞争"],
    "复联5谁回归": ["秋山"],
    "马戏盛典冠军": ["向日葵"],
}


def _overlaps(seg_start, seg_end, req_start, req_end):
    return seg_start < req_end and seg_end > req_start


def evaluate(top_k=5, media_id=None, verbose=False, mode="hybrid+rerank", quiet=False):
    rows = []
    for item in GOLDEN:
        try:
            hits = retrieval_search(item["goal"], limit=top_k, media_id=media_id, mode=mode)
        except Exception as e:
            # 检索整条挂了：这题记为全错，评测不中断
            print("[warn] %s -> %s: %s" % (item["name"], type(e).__name__, e))
            hits = []

        covered = set()
        first_rank = None
        for rank, h in enumerate(hits, 1):
            for j, (rs, re_) in enumerate(item["ranges"]):
                if _overlaps(h["start"], h["end"], rs, re_):
                    covered.add(j)
                    if first_rank is None:
                        first_rank = rank

        rows.append({
            "name": item["name"],
            "goal": item["goal"],
            "hit": bool(covered),
            "recall": len(covered) / len(item["ranges"]),
            "mrr": (1.0 / first_rank) if first_rank else 0.0,
            "first_rank": first_rank,
        })

        if verbose:
            print("\n[%s] %s" % (item["name"], item["goal"]))
            for rank, h in enumerate(hits, 1):
                sec = h["start"] // 1000
                mark = ""
                for rs, re_ in item["ranges"]:
                    if _overlaps(h["start"], h["end"], rs, re_):
                        mark = " ✔"
                print("  #%d [%02d:%02d] (%s) %s%s"
                      % (rank, sec // 60, sec % 60, h["source"], h["text"][:34], mark))

    n = len(rows)
    summary = {
        "mode": mode,
        "hitRate": sum(1 for r in rows if r["hit"]) / n,
        "avgRecall": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "rows": rows,
    }

    if not quiet:
        print("\n" + "=" * 66)
        print("检索层评测（%d 题，top=%d，media_id=%s，mode=%s）"
              % (n, top_k, media_id, mode))
        print("=" * 66)
        print("%-22s %-7s %-9s %-6s" % ("题目", "hit@K", "recall@K", "MRR"))
        for r in rows:
            print("%-22s %-7s %-9.2f %-6.3f"
                  % (r["name"], "✔" if r["hit"] else "✘", r["recall"], r["mrr"]))
        print("-" * 66)
        print("汇总：hit@%-2d = %.1f%%   平均 recall = %.2f   MRR = %.3f"
              % (top_k, summary["hitRate"] * 100, summary["avgRecall"], summary["mrr"]))
    return summary


def _warmup(media_id=None):
    """先跑一次检索，把 embedding 客户端、jieba 词典、Qdrant 连接热起来。

    不预热的话，第一个被测配置会额外背几百毫秒的冷启动，
    「融合比纯向量快」这种假结论就是这么来的。
    """
    try:
        retrieval_search("预热查询", limit=1, media_id=media_id)
    except Exception:
        pass


def _median_ms(ms):
    """取中位数而不是平均：第一次调用要背 DNS/TLS 建连，几秒的离群值会把均值拉飞。"""
    if not ms:
        return 0.0
    xs = sorted(ms)
    return xs[len(xs) // 2]


def _rank_at(hits, ranges, k):
    """在已排好序的 hits 上截前 k 条，算 hit / MRR / recall。"""
    covered = set()
    first = None
    for rank, h in enumerate(hits[:k], 1):
        for j, (rs, re_) in enumerate(ranges):
            if _overlaps(h["start"], h["end"], rs, re_):
                covered.add(j)
                if first is None:
                    first = rank
    return bool(covered), (1.0 / first if first else 0.0), len(covered) / len(ranges)


def compare(top_k=5, media_id=None):
    """三种配置 × 三个 K：同一套题、同一批数据，看每层到底涨了多少。

    为什么要一次跑多个 K：题少的时候只看 top5 会全部饱和（8 题时三档都是
    100%），看起来"融合没用"。K 越小越能暴露排序问题，K 越大越接近
    "到底有没有召回"。两个口径一起给，才不会被单个 K 骗。

    为什么每档都标耗时：检索层的收益不是白来的，rerank 是跨编码调用，
    要能和"涨了几个点"放在一起看值不值。
    """
    ks = sorted({k for k in (1, 3, top_k) if k <= top_k})
    n = len(GOLDEN)
    _warmup(media_id)
    print("=" * 78)
    print("检索配置对比（%d 题，recall 深度=%d，media_id=%s）"
          % (n, top_k, media_id))
    print("=" * 78)
    print("%-22s %s %-9s %-8s" % (
        "配置", "".join("hit@%-6d" % k for k in ks), "MRR@%d" % top_k, "平均耗时"))
    print("-" * 78)

    results = {}
    for mode, label in MODES:
        hit = {k: 0 for k in ks}
        mrr = 0.0
        ms = []
        for item in GOLDEN:
            t0 = time.perf_counter()
            try:
                hits = retrieval_search(item["goal"], limit=top_k,
                                        media_id=media_id, mode=mode)
            except Exception as e:
                print("[warn] %s -> %s: %s" % (item["name"], type(e).__name__, e))
                hits = []
            ms.append((time.perf_counter() - t0) * 1000)
            for k in ks:
                ok, r, _ = _rank_at(hits, item["ranges"], k)
                hit[k] += ok
                if k == top_k:
                    mrr += r
        row = {
            "mode": mode,
            "hit": {k: hit[k] / n for k in ks},
            "mrr": mrr / n,
            "avgMs": _median_ms(ms),
        }
        results[mode] = row
        print("%-22s %s %-9.3f %-8s" % (
            label,
            "".join("%-10s" % ("%.1f%%" % (100 * row["hit"][k])) for k in ks),
            row["mrr"], "%.0fms" % row["avgMs"]))

    base = results["vector"]
    final = results["hybrid+rerank"]
    print("-" * 78)
    print("相对纯向量：hit@1 %+.1f 个点，hit@%d %+.1f 个点，MRR %+.3f"
          % ((final["hit"][1] - base["hit"][1]) * 100,
             top_k, (final["hit"][top_k] - base["hit"][top_k]) * 100,
             final["mrr"] - base["mrr"]))
    return results


def content_eval(top_k=5, media_id=None, mode="hybrid+rerank"):
    """内容级评测：扩窗到底有没有把答案带进证据里。

    为什么要单独评：hit@K 只看时间区间有没有被命中。扩窗会把 k 覆盖的区间
    撑大到 28 秒，几乎什么都能重叠上，指标全部饱和，扩窗的收益在名次指标上
    根本测不出来。但扩窗要解决的恰恰是「命中了结论那半句、宾语在下一句」，
    那就别绕弯子：直接检查返回文本里有没有答案关键词。

    两组对照：不扩窗（句子级）vs 扩窗（窗口），同一批题、同一批候选。
    """
    items = [x for x in GOLDEN if x["name"] in ANSWER_KEYS]
    print("=" * 72)
    print("内容级评测：答案关键词是否进入证据（%d 题，mode=%s）" % (len(items), mode))
    print("=" * 72)
    print("%-14s %-14s %-14s %-10s" % ("证据形态", "答案覆盖率", "完全命中", "平均耗时"))
    print("-" * 72)
    _warmup(media_id)

    out = {}
    for windowed in (False, True):
        fn = retrieval_search_windowed if windowed else retrieval_search
        cov_sum = 0.0
        full = 0
        ms = []
        for item in items:
            keys = ANSWER_KEYS[item["name"]]
            t0 = time.perf_counter()
            try:
                hits = fn(item["goal"], limit=top_k, media_id=media_id, mode=mode)
            except Exception as e:
                print("[warn] %s -> %s" % (item["name"], type(e).__name__))
                hits = []
            ms.append((time.perf_counter() - t0) * 1000)
            text = simplify_text(" ".join(h.get("text") or "" for h in hits))
            got = sum(1 for kw in keys if simplify_text(kw) in text)
            cov_sum += got / len(keys)
            full += (got == len(keys))
        out[windowed] = {
            "coverage": cov_sum / len(items),
            "full": full / len(items),
            "avgMs": _median_ms(ms),
        }
        print("%-14s %-14s %-14s %-10s" % (
            "是（窗口）" if windowed else "否（句子级）",
            "%.1f%%" % (100 * out[windowed]["coverage"]),
            "%.1f%%" % (100 * out[windowed]["full"]),
            "%.0fms" % out[windowed]["avgMs"]))
    print("-" * 72)
    print("扩窗收益：答案覆盖率 %+.1f 个点，完全命中 %+.1f 个点"
          % ((out[True]["coverage"] - out[False]["coverage"]) * 100,
             (out[True]["full"] - out[False]["full"]) * 100))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--media-id", type=int, default=None)
    ap.add_argument("--mode", default="hybrid+rerank", choices=["vector", "hybrid", "hybrid+rerank"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--content", action="store_true", help="内容级评测：扩窗前后的答案覆盖率")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.content:
        content_eval(top_k=args.top_k, media_id=args.media_id, mode=args.mode)
    elif args.compare:
        compare(top_k=args.top_k, media_id=args.media_id)
    else:
        evaluate(top_k=args.top_k, media_id=args.media_id,
                 mode=args.mode, verbose=args.verbose)
