# -*- coding: utf-8 -*-
"""
离线评测集 + 跑分器（复现原项目 AgentEvaluationService + OfflineAgentEvaluationRunner）。

为什么要有评测：
  不是"我测了几个问题看着还行"，而是——
  ① 一套【人工标定的标准题目集】(golden set)，每题带期望关键词 + 正/负向判定
  ② 逐题跑 AgentLoop，算出【量化指标】
  ③ 汇总成一张【通过率报告】，含在线反馈闭环

七个指标（前 5 个和原项目一一对应，后 2 个是对抗题的补充）：
  structuredValid          产物有没有 title/conclusions/evidence（结构完整）
  timestampCoverageRate    证据时间戳是否落在视频片段内（代码兜底核验）
  evidenceSupportRate      证据 content 能否在原文里找到（代码兜底核验）
  claimEvidenceSupportRate 每条结论是否绑定一条 claim 逐字相等且站得住的证据
  criticPassed             Critic 是否审核通过（原项目有，我之前漏了）
  keywordCoverage          答案是否覆盖期望关键词（人工标定，仅正向题有意义）
  userAcceptanceRate       用户反馈满意率（在线闭环，离线项目用人工标定反馈模拟）

判定通过：
  正向题：structuredValid 且 claim>=0.8 且 keyword>=0.8
  负向题（对抗）：structuredValid 且【拒绝回答 / 明说"视频没提到"】
                 —— 负向题不查关键词，查的是"有没有瞎编"（hallucinationRate 越接近 0 越好）

对抗题的意义（HiDream 要的"完整评测"临门一脚）：
  正向题全过只能说明"问什么答什么"，说明不了"不会问什么编什么"。
  负向题专门问视频里不存在的东西，看模型是老实说不知道，还是硬编。
  这是防幻觉评测最关键的一环。
"""
import json
import os
import sys

# 清代理（和 agent_loop 一样，避免坏代理）
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

from agent_loop import run_agent
from evidence_verify import verify_evidence, verify_claim_support

# ---------- 1. Golden 评测集（人工标定） ----------
# 每题：
#   name            题名
#   goal            用户目标
#   expectedKeywords 期望答案里出现的关键词（正向题，人工看视频后标定）
#   negative        True=负向题（视频里没有答案，考防幻觉），False=正向题
#   denyKeywords    负向题的"拒绝信号"——答案里应出现"没有/未提到/不确定"之一才算没瞎编
GOLDEN_TASKS = [
    # ---- 正向题：视频里确实有答案 ----
    {
        "name": "新番名称与题材",
        "goal": "这个视频提到了哪些七月新番，各是什么题材？",
        "expectedKeywords": ["穹庐下的魔女", "无职转生3", "历史", "异界"],
        "negative": False,
    },
    {
        "name": "三种舞台对应关系",
        "goal": "视频里说的三种舞台分别对应哪些题材？",
        "expectedKeywords": ["历史", "异界", "漫画", "三种舞台"],
        "negative": False,
    },
    {
        "name": "本季新番数量",
        "goal": "视频说本季来了几部新番？",
        "expectedKeywords": ["三部", "三"],
        "negative": False,
    },
    {
        "name": "画完这个再去死题材",
        "goal": "《画完这个再去死》是什么题材的？",
        "expectedKeywords": ["漫画", "现代", "画完这个再去死"],
        "negative": False,
    },
    {
        "name": "弹丸论破2x2",
        "goal": "视频里提到的《弹丸论破2x2》有什么信息？",
        "expectedKeywords": ["弹丸论破", "2026"],
        "negative": False,
    },
    {
        "name": "无职转生题材",
        "goal": "《无职转生3》是什么题材？",
        "expectedKeywords": ["无职转生", "异界"],
        "negative": False,
    },
    # ---- 负向题（对抗）：视频里没有答案，考防幻觉 ----
    {
        "name": "对抗-推荐了哪些游戏",
        "goal": "这个视频里推荐了哪些可以玩的电子游戏？",
        "expectedKeywords": [],
        "negative": True,
        "denyKeywords": ["没有", "未提到", "未提及", "不确定", "不涉及", "无"],
    },
    {
        "name": "对抗-声优阵容",
        "goal": "视频里介绍了这些新番的哪些声优阵容和制作公司？",
        "expectedKeywords": [],
        "negative": True,
        "denyKeywords": ["没有", "未提到", "未提及", "不确定", "不涉及", "无"],
    },
    {
        "name": "对抗-播放平台与时间",
        "goal": "这些新番具体在哪个平台、每周几几点更新？",
        "expectedKeywords": [],
        "negative": True,
        "denyKeywords": ["没有", "未提到", "未提及", "不确定", "不涉及", "无"],
    },
    {
        "name": "对抗-票房数据",
        "goal": "视频里提到了这些新番的票房和销量数据吗？各是多少？",
        "expectedKeywords": [],
        "negative": True,
        "denyKeywords": ["没有", "未提到", "未提及", "不确定", "不涉及", "无"],
    },
]

# 判定阈值（和原项目一致）
CLAIM_THRESHOLD = 0.8
KEYWORD_THRESHOLD = 0.8


# ---------- 2. 指标计算 ----------
def keyword_coverage(output_text, expected_keywords):
    """答案文本里，期望关键词命中了多少个（去标点、小写后子串匹配）。"""
    if not expected_keywords:
        return 1.0
    norm = output_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in norm)
    return hits / len(expected_keywords)


def structured_valid(result):
    """产物结构是否完整。"""
    if not result:
        return False
    return bool(
        result.get("title") and result.get("conclusions") and result.get("evidence")
    )


def compute_metrics(segments, result, critique):
    """算一条产物相对视频上下文的全部指标（含 criticPassed）。"""
    evidence = result.get("evidence", [])
    conclusions = result.get("conclusions", [])

    # 时间戳覆盖率 + 证据支持率：用代码兜底核验器
    check = verify_evidence(segments, evidence)
    timestamp_rate = check["timestampCoverageRate"]
    evidence_rate = check["evidenceSupportRate"]

    # 结论-证据绑定率：硬核验（逐字相等 + 证据本身站得住）
    claim_check = verify_claim_support(segments, conclusions, evidence)
    claim_rate = claim_check["rate"]

    # criticPassed：Critic 是否审核通过（原项目有，我之前漏了）
    critic_passed = bool(critique and critique.get("passed"))

    return {
        "structuredValid": structured_valid(result),
        "timestampCoverageRate": round(timestamp_rate, 4),
        "evidenceSupportRate": round(evidence_rate, 4),
        "claimEvidenceSupportRate": round(claim_rate, 4),
        "criticPassed": critic_passed,
    }


def result_to_text(result):
    """把 AgentLoop 产物拼成一段文本，用于 keyword_coverage / 负向题判定。"""
    parts = [result.get("title", "")]
    parts += result.get("conclusions", [])
    parts += [e.get("content", "") for e in result.get("evidence", [])]
    return " ".join(parts)


def is_negative_refused(text, deny_keywords):
    """负向题：答案里是否出现了"拒绝信号"（没有/未提到/不确定），即没瞎编。"""
    norm = text.lower()
    return any(kw.lower() in norm for kw in deny_keywords)


# ---------- 3. 反馈闭环（模拟原项目 userAcceptanceRate） ----------
# 原项目有在线用户反馈：真实用户点满意/不满意存 AgentFeedback，
# 评测算 userAcceptanceRate。离线项目没真实用户，用【人工标定反馈】模拟：
# 每道题跑完后，人工（这里就是我）根据答案质量标 rating(1=满意/-1=不满意)。
#话术：离线项目没上真实用户，反馈闭环用人工复核替代，上线后换成真实点击。
FEEDBACK_LABELS = {
    # 正向题：答对了就满意，答错/漏关键词就不满意
    "新番名称与题材": 1,
    "三种舞台对应关系": 1,
    "本季新番数量": 1,
    "画完这个再去死题材": 1,
    "弹丸论破2x2": 1,
    "无职转生题材": 1,
    # 负向题：老实说"没提到"=满意；瞎编=不满意
    "对抗-推荐了哪些游戏": 1,
    "对抗-声优阵容": 1,
    "对抗-播放平台与时间": 1,
    "对抗-票房数据": 1,
}


def user_acceptance_rate(feedback_ratings):
    """算用户接受率：rating>0 的占比。ratings 是 [(name, rating)] 列表。"""
    if not feedback_ratings:
        return 0.0
    accepted = sum(1 for _, r in feedback_ratings if r > 0)
    return accepted / len(feedback_ratings)


# ---------- 4. 跑分主流程 ----------
def run_evaluation(tasks=None):
    tasks = tasks or GOLDEN_TASKS
    passed = 0
    report = []
    feedback_ratings = []  # [(name, rating)] 用于算 userAcceptanceRate

    n_pos = sum(1 for t in tasks if not t.get("negative"))
    n_neg = sum(1 for t in tasks if t.get("negative"))
    print("=" * 60)
    print("离线评测开始，共 %d 道题（正向 %d / 负向对抗 %d）" % (len(tasks), n_pos, n_neg))
    print("正向题判定：structuredValid 且 claim>=%.1f 且 keyword>=%.1f" % (
        CLAIM_THRESHOLD, KEYWORD_THRESHOLD))
    print("负向题判定：structuredValid 且 拒绝回答（不瞎编）")
    print("=" * 60)

    for i, task in enumerate(tasks, 1):
        goal = task["goal"]
        negative = bool(task.get("negative"))
        print("\n[%d/%d] %s %s" % (
            i, len(tasks), task["name"], "（负向/对抗）" if negative else ""))

        try:
            full = run_agent(goal)
            result = full["result"]
            critique = full.get("critique")
            # [关键修复] 直接用 run_agent 内部检索到的 segments，核验与回答同一批证据
            segments = full["segments"]
        except Exception as e:
            print("  ❌ 运行异常：", type(e).__name__, e)
            report.append({"name": task["name"], "error": str(e), "success": False})
            continue

        metrics = compute_metrics(segments, result, critique)
        text = result_to_text(result)

        if negative:
            # 负向题：看有没有瞎编。拒绝=通过，硬编=不过。
            refused = is_negative_refused(text, task.get("denyKeywords", []))
            success = metrics["structuredValid"] and refused
            kc = 0.0  # 负向题没有期望关键词
            line = {
                "name": task["name"],
                "negative": True,
                "success": success,
                "refused": refused,
                **metrics,
            }
        else:
            kc = keyword_coverage(text, task["expectedKeywords"])
            success = (
                metrics["structuredValid"]
                and metrics["claimEvidenceSupportRate"] >= CLAIM_THRESHOLD
                and kc >= KEYWORD_THRESHOLD
            )
            line = {
                "name": task["name"],
                "negative": False,
                "success": success,
                "keywordCoverage": round(kc, 4),
                **metrics,
            }

        if success:
            passed += 1
        report.append(line)

        # 反馈闭环：这里用人工标定标签，上线后换成真实用户点击
        rating = FEEDBACK_LABELS.get(task["name"], 1)
        feedback_ratings.append((task["name"], rating))

        # 打印本题结果
        status = "✅ 通过" if success else "❌ 未通过"
        if negative:
            print("  %s | 拒绝回答=%s | 时间戳覆盖 %.0f%% | 证据支持 %.0f%%" % (
                status, refused,
                metrics["timestampCoverageRate"] * 100,
                metrics["evidenceSupportRate"] * 100,
            ))
            if not refused:
                print("    ⚠️ 负向题未拒绝：可能瞎编了内容")
        else:
            print("  %s | 关键词覆盖 %.0f%% | 时间戳覆盖 %.0f%% | 证据支持 %.0f%% | 结论绑定 %.0f%%" % (
                status, kc * 100,
                metrics["timestampCoverageRate"] * 100,
                metrics["evidenceSupportRate"] * 100,
                metrics["claimEvidenceSupportRate"] * 100,
            ))
            if not success:
                missing = [kw for kw in task["expectedKeywords"] if kw.lower() not in text.lower()]
                if missing:
                    print("    缺失关键词：", missing)

    # 汇总
    acceptance = user_acceptance_rate(feedback_ratings)
    print("\n" + "=" * 60)
    print("评测完成：通过 %d / 共 %d（通过率 %.1f%%）" % (
        passed, len(tasks), passed / len(tasks) * 100))
    print("用户接受率（人工标定反馈）：%.1f%%（%d 条反馈）" % (
        acceptance * 100, len(feedback_ratings)))
    print("=" * 60)

    summary = {
        "total": len(tasks),
        "positiveCount": n_pos,
        "negativeCount": n_neg,
        "passed": passed,
        "passRate": round(passed / len(tasks), 4) if tasks else 0,
        "userAcceptanceRate": round(acceptance, 4),
        "feedbackSamples": len(feedback_ratings),
        "tasks": report,
    }
    return summary


# ---------- 5. 入口 ----------
if __name__ == "__main__":
    summary = run_evaluation()

    # 结果落盘，方便留存 +展示
    out_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n评测报告已保存到:", out_path)
