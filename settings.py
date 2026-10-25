# -*- coding: utf-8 -*-
"""配置读取：环境变量优先，其次项目根目录的 .env。

为什么单独抽一个文件：
    改造前 9 个模块各自写死一个项目外的 .env 绝对路径。
    换台机器、换目录、或者把那个目录挪走，读到的就是空字符串，
    embedding / rerank 会静默失败并退化成 BM25，而 /health 仍然报 up。
    配置只留一个入口，出问题才查得到。

优先级：os.environ > 项目内 .env > 默认值。
    Web 服务由 app.py 调用 load_env() 把 .env 灌进环境变量；
    CLI 脚本（video_to_text_local.py 等）直接 get() 也能读到同一个文件。
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE, ".env")

_cache = {"mtime": None, "data": {}}


def _parse(path):
    """解析 .env 成 dict；文件不存在或读不了返回空 dict（带 mtime 缓存）。"""
    try:
        st = os.stat(path)
    except OSError:
        return {}
    if _cache["mtime"] == st.st_mtime:
        return _cache["data"]

    data = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        return {}

    _cache["mtime"] = st.st_mtime
    _cache["data"] = data
    return data


def load_env(path=None):
    """把 .env 写进 os.environ（已存在的环境变量优先）。返回是否读到文件。"""
    path = path or ENV_FILE
    if not os.path.exists(path):
        return False
    for k, v in _parse(path).items():
        if v and k not in os.environ:
            os.environ[k] = v
    return True


def get(key, default=""):
    """读一个配置：环境变量 > .env > default。"""
    return os.environ.get(key) or _parse(ENV_FILE).get(key) or default


def key_status():
    """给 /health 用：关键 key 有没有读到。只报布尔，不返回值本身。"""
    return {k: bool(get(k)) for k in ("DASHSCOPE_KEY", "DEEPSEEK_KEY")}


def qdrant_kwargs():
    """Qdrant 客户端参数。

    本地跑是 127.0.0.1:6333；进了 compose 网络就得写服务名 qdrant，
    所以地址不能硬编码——否则容器里连的是容器自己的 127.0.0.1。
    """
    return {
        "host": get("QDRANT_HOST", "127.0.0.1"),
        "port": int(get("QDRANT_PORT", "6333")),
    }
