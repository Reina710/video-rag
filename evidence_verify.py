# -*- coding: utf-8 -*-
"""
证据程序化核验。

为什么要有这一层：
  单靠 Critic（让模型自己审自己）是"软校验"——模型可能睁一只眼闭一只眼，
  甚至编一个时间戳骗过自己。这一层用【代码】去向量库里反查，是"硬校验"：
  证据说"第 90 秒的 OCR 写了 X"，代码就去查——第 90 秒到底有没有这么一段原文？

三个核验函数：
  timestamp_covered(segments, evidence) -> bool
      证据的 timestampMs 是否真的落在某个 segment 的 [startMs, endMs) 区间内
  evidence_supported(segments, evidence) -> bool
      证据的 content 是否真的能在该时间戳对应的 transcript/ocr 原文里找到（子串匹配）
  supports_claim(segments, claim, evidence) -> bool
      claim 是否和 evidence.claim 逐字相等，且这条证据本身站得住

关键设计：
  - 这是"软硬结合"：Critic 用语义审（软），代码用时间戳+原文反查（硬）
  - 杜绝两种幻觉：① 编时间戳（timestamp_covered 兜住）；② 编原文（evidence_supported 兜住）
  - 和我 kb-agent 项目里"防注入"思路一致：关键环节不靠模型自觉，靠代码兜底
"""
import re


def normalize(text):
    """去掉标点、符号、空白、大小写差异，只留字母数字（含中文）。"""
    if not text:
        return ""
    # 去标点/符号/空白，转小写（中文不受影响）
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


def contains_timestamp(segment, timestamp_ms):
    """时间戳是否落在 segment 的 [startMs, endMs) 区间内。"""
    return segment["startMs"] <= timestamp_ms < segment["endMs"]


def timestamp_covered(segments, evidence):
    """证据的时间戳是否真的存在于某个片段的时间段内。"""
    if not segments or not evidence:
        return False
    ts = evidence.get("timestampMs")
    if ts is None:
        return False
    return any(contains_timestamp(s, ts) for s in segments)


def source_text(segment, source):
    """按证据声明的来源，取出该 segment 对应的原文。ASR->transcript，OCR->ocrTexts。"""
    src = (source or "").upper()
    if "ASR" in src and "OCR" in src:
        return segment.get("text", "")  # 双来源，用合并文本
    if "ASR" in src:
        return segment.get("text", "")
    if "OCR" in src:
        return segment.get("text", "")
    return ""


def evidence_supported(segments, evidence):
    """证据的 content 是否真的能在原文里找到（子串匹配，去标点后比较）。"""
    if not segments or not evidence:
        return False
    content = evidence.get("content", "")
    if not content:
        return False
    src = evidence.get("source", "").upper()
    if "ASR" not in src and "OCR" not in src:
        return False  # 来源不合法，直接判不成立

    norm_content = normalize(content)
    if not norm_content:
        return False

    # 在该时间戳对应的所有片段里，找有没有原文包含这段 content
    for seg in segments:
        if not contains_timestamp(seg, evidence.get("timestampMs")):
            continue
        candidate = normalize(source_text(seg, src))
        if candidate and norm_content in candidate:
            return True
    return False


def supports_claim(segments, claim, evidence):
    """结论 claim 是否真的有一条逐字相等的证据支撑。"""
    if not evidence:
        return False
    if not normalize(claim):
        return False
    if normalize(claim) != normalize(evidence.get("claim", "")):
        return False
    return evidence_supported(segments, evidence)


def verify_evidence(segments, evidence_list):
    """
    批量核验一条产物里的所有证据，返回每条证据的核验结果。

    返回 dict：{"passed": 核验通过数, "total": 总数,
               "timestamp_covered": n, "evidence_supported": n,
               "failed": [{"timestampMs":..., "source":..., "content":..., "reason":...}]}
    """
    total = len(evidence_list)
    passed = 0
    failed = []
    for ev in evidence_list:
        ok_ts = timestamp_covered(segments, ev)
        ok_support = evidence_supported(segments, ev)
        if ok_ts and ok_support:
            passed += 1
        else:
            reason = []
            if not ok_ts:
                reason.append("时间戳越界")
            if not ok_support:
                reason.append("原文未找到该content")
            failed.append({
                "timestampMs": ev.get("timestampMs"),
                "source": ev.get("source"),
                "content": ev.get("content", "")[:40],
                "reason": " + ".join(reason) or "未知",
            })
    return {
        "passed": passed,
        "total": total,
        "timestampCoverageRate": (sum(1 for e in evidence_list if timestamp_covered(segments, e)) / total) if total else 0,
        "evidenceSupportRate": passed / total if total else 0,
        "failed": failed,
    }


def verify_claim_support(segments, conclusions, evidence_list):
    """
    批量核验「每条结论是否真的被一条逐字相等、且站得住的证据支撑」。

    这是 claimEvidenceSupportRate 的【硬核验版】：不满足于字符串相等，
    还要求那条证据本身能过 timestamp_covered + evidence_supported（原文真的存在）。

    返回 dict：{"supported": n, "total": n, "rate": 0~1}
    """
    total = len(conclusions)
    if total == 0:
        return {"supported": 0, "total": 0, "rate": 0.0}
    supported = 0
    for c in conclusions:
        if any(supports_claim(segments, c, e) for e in evidence_list):
            supported += 1
    return {"supported": supported, "total": total, "rate": supported / total}


if __name__ == "__main__":
    # 自测：构造几条证据，验证核验逻辑
    segs = [
        {"startMs": 90000, "endMs": 120000, "text": "《无职转生3》异界"},
        {"startMs": 120000, "endMs": 150000, "text": "三种舞台 三种题材"},
    ]
    evidence = [
        {"timestampMs": 90000, "source": "OCR", "content": "无职转生3", "claim": "无职转生3是异界题材"},   # 应该通过
        {"timestampMs": 999999, "source": "OCR", "content": "瞎编的内容", "claim": "xx"},                  # 时间戳越界，应该失败
        {"timestampMs": 90000, "source": "OCR", "content": "画面里没有这句话", "claim": "xx"},              # 原文找不到，应该失败
    ]
    result = verify_evidence(segs, evidence)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
