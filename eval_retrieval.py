# -*- coding: utf-8 -*-
"""
检索层评测：只评「检索有没有捞到正确的证据」，不评生成。

为什么要单独评检索（面试核心）：
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
    python eval_retrieval.py --compare                # 三种配置横向对比
    python eval_retrieval.py --mode vector --top-k 5
    python eval_retrieval.py --media-id 5 --verbose
"""
import argparse
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

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
]

MODES = [
    ("vector", "① 纯向量"),
    ("hybrid", "② 向量 + BM25 + RRF"),
    ("hybrid+rerank", "③ 再加 rerank"),
]


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


def compare(top_k=5, media_id=None):
    """三种配置横向对比：用同一套题、同一批数据，看每层到底涨了多少。"""
    print("=" * 70)
    print("检索配置对比（%d 题，top=%d，media_id=%s）" % (len(GOLDEN), top_k, media_id))
    print("=" * 70)
    print("%-22s %-12s %-12s %-10s" % ("配置", "hit@K", "recall@K", "MRR"))
    print("-" * 70)
    results = {}
    for mode, label in MODES:
        r = evaluate(top_k=top_k, media_id=media_id, mode=mode, quiet=True)
        results[mode] = r
        print("%-22s %-12s %-12.2f %-10.3f"
              % (label, "%.1f%%" % (r["hitRate"] * 100), r["avgRecall"], r["mrr"]))
    base = results["vector"]
    final = results["hybrid+rerank"]
    print("-" * 70)
    print("相对纯向量：hit@%d %+.1f 个百分点，MRR %+.3f"
          % (top_k, (final["hitRate"] - base["hitRate"]) * 100, final["mrr"] - base["mrr"]))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--media-id", type=int, default=None)
    ap.add_argument("--mode", default="hybrid+rerank", choices=["vector", "hybrid", "hybrid+rerank"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare(top_k=args.top_k, media_id=args.media_id)
    else:
        evaluate(top_k=args.top_k, media_id=args.media_id,
                 mode=args.mode, verbose=args.verbose)
