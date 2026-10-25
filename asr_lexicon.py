# -*- coding: utf-8 -*-
"""ASR 词表：热词（喂给模型做偏置）+ 纠错（改输出文本）。

为什么是两张表而不是一张：
    热词是「偏置」——转写时告诉模型"注意这些词"，但 small/int8 的模型不一定听话；
    纠错是「兜底」——输出已经错了，按已知错法替换回来。
    只做热词，错字照旧；只做纠错，得先知道错成什么样。两层一起才稳。

文件格式（都在 data/ 下）：
    hotwords.txt     一行一个词，# 开头是注释
    asr_fixups.txt   一行一条「错=对」，# 开头是注释

覆盖方式（换领域时不用改代码）：
    ASR_HOTWORDS="词1 词2"   覆盖热词表
    ASR_FIXUPS=0            关掉纠错（排查是模型错还是纠错表改错时用）

注意：热词只影响【新上传的转写】。已经索引过的视频走的是上传去重，不会重跑 ASR。
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
HOTWORDS_FILE = os.path.join(DATA_DIR, "hotwords.txt")
FIXUPS_FILE = os.path.join(DATA_DIR, "asr_fixups.txt")

# 热词太多会把 prompt 挤满、反而干扰解码，30 个是经验上限
MAX_HOTWORDS = 30

_cache = {"key": None, "hot": "", "fix": []}


def _read(path):
    """读非注释行。文件不存在就返回空列表（热词不是必需品，缺了照常转写）。"""
    try:
        raw = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return []
    out = []
    for line in raw:
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _stamp():
    """文件 mtime + 两个环境变量，任何一个变了就重新加载。"""
    mtimes = []
    for p in (HOTWORDS_FILE, FIXUPS_FILE):
        try:
            mtimes.append(os.stat(p).st_mtime)
        except OSError:
            mtimes.append(0)
    return (tuple(mtimes),
            os.environ.get("ASR_HOTWORDS", ""),
            os.environ.get("ASR_FIXUPS", ""))


def _parse_fixups(lines):
    """解析「错=对」。长的错词先替换，避免短的先命中把长的拆坏。"""
    pairs = []
    for line in lines:
        if "=" not in line:
            continue
        wrong, right = line.split("=", 1)
        wrong, right = wrong.strip(), right.strip()
        if wrong and right and wrong != right:
            pairs.append((wrong, right))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def _ensure():
    key = _stamp()
    if _cache["key"] != key:
        env_hot = os.environ.get("ASR_HOTWORDS", "").replace("，", " ").replace(",", " ").split()
        words = env_hot or _read(HOTWORDS_FILE)
        _cache["hot"] = " ".join(words[:MAX_HOTWORDS])
        _cache["fix"] = _parse_fixups(_read(FIXUPS_FILE))
        _cache["key"] = key
    return _cache


def hotwords():
    """给 faster-whisper 的 hotwords 参数。空表返回 None（传空串会变成空 prompt）。"""
    return _ensure()["hot"].strip() or None


def fixups():
    """当前生效的纠错表，测试和排查用。"""
    return list(_ensure()["fix"])


def correct(text):
    """按错字表替换。ASR_FIXUPS=0 时原样返回。"""
    if os.environ.get("ASR_FIXUPS", "1").lower() in ("0", "false", "off"):
        return text or ""
    s = text or ""
    for wrong, right in _ensure()["fix"]:
        if wrong in s:
            s = s.replace(wrong, right)
    return s
