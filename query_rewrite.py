# -*- coding: utf-8 -*-
"""
查询改写：把多轮追问改写成不依赖上下文的完整问题。

为什么需要（真实场景）：
    用户问完「本季高分作有哪几部？」，接着问「那第二部呢」。
    这句话单独送进检索几乎必挂 —— 它没有实体，向量和 BM25 都不知道在问什么。
    改写之后再检索，才能捞到《无职转生3》那一段。

和校园项目的关系（面试主线）：
    同一个思路：追问 + 最近历史 → 补全成完整问题再检索。
    视频这边多一层"时间指代"（"刚才那段"），但改写方式一样。

工程约定：
    1. 没有历史 / 问题已经自足 → 原样返回，不花那一次 LLM
    2. 改写失败 → 返回原问题（宁可检索差一点，也不让链路断）
    3. 只输出改写结果，不做解释
"""
import os
import re

import settings

# 出现这些词，才认为"可能是追问"
_FOLLOWUP_HINTS = (
    "它", "他", "她", "这", "那", "上面", "刚才", "前面",
    "第二", "第三", "还有", "呢", "哪个", "那段", "那里",
)


def _env_value(key):
    """读配置：环境变量优先，其次项目内 .env。"""
    return settings.get(key)


def need_rewrite(question, history):
    """判断要不要改写。宁可少改，也别把自足的问题改坏。"""
    if not history:
        return False
    q = (question or "").strip()
    if not q:
        return False
    # 够长且没有指代词 → 认为自足，不改
    if len(q) >= 18 and not any(h in q for h in _FOLLOWUP_HINTS):
        return False
    return any(h in q for h in _FOLLOWUP_HINTS) or len(q) < 8


def rewrite(question, history):
    """返回 (检索用的问题, 是否真的改写了)。失败时原样返回并标记 False。"""
    q = (question or "").strip()
    if not need_rewrite(q, history):
        return q, False

    try:
        from openai import OpenAI
    except ImportError:
        return q, False

    client = OpenAI(
        api_key=_env_value("DEEPSEEK_KEY"),
        base_url="https://api.deepseek.com",
        timeout=30,
        max_retries=2,
    )

    # 只取最近 2 轮：太长会把无关信息带进来，反而改歪
    lines = []
    for m in history[-4:]:
        role = "用户" if m.get("role") == "user" else "助手"
        text = (m.get("text") or "")[:200]
        lines.append("%s：%s" % (role, text))

    prompt = (
        "你是检索查询改写器。根据对话历史，把用户这一轮的追问改写成"
        "不依赖上下文的完整问题，供向量检索使用。\n"
        "规则：\n"
        "1) 只输出改写后的问句本身，不要解释、不要加引号；\n"
        "2) 补全指代词（它/这个/那部/第二部/刚才那段）指的具体对象；\n"
        "3) 不要引入历史里没有的信息；\n"
        "4) 如果原问题已经完整，原样输出。\n\n"
        "对话历史：\n" + "\n".join(lines) +
        "\n\n本轮问题：" + q + "\n改写结果："
    )

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip()
        out = out.strip("「」\"'“”")
        out = re.sub(r"^改写结果[:：]\s*", "", out)
        if not out or len(out) > 200:
            return q, False
        return out, out != q
    except Exception:
        # 改写是增强项，不是命脉：挂了就用原问题
        return q, False
