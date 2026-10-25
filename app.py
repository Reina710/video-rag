# -*- coding: utf-8 -*-
"""
第四步：把三步串成一个 Web 服务（Flask 后端）。

参考 DOVideo-AI 的接口设计（上传和问答分开），但砍掉鉴权/异步/多模式，
第一版只留最核心的 /ask：问问题，返回带时间戳的回答。

接口：
  GET  /           首页（简单说明）
  POST /ask        入参 {"question": "..."}，返回 {"answer": "...", "hits": [...]}

面试可讲的点（[面试] 标注）：
  - 把三个命令行脚本封装成 REST 服务，是"脚本 -> 产品"的关键一跳
  - /ask 返回的 hits 里带时间戳，前端可以渲染"点击跳转到视频第几秒"
"""
import json
import io
import os
import time
import uuid
from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context
from openai import OpenAI
from qdrant_client import QdrantClient

# AgentLoop 的三个角色 + 主循环（Planner/Executor/Critic），独立在 agent_loop.py
from agent_loop import run_agent
from minio import Minio

# [面试] 落库层：SQLite/MySQL 双后端 + 用户/视频/任务/反馈，异步写（线程队列，对应 RocketMQ 削峰）
import storage
# [面试] 自动解析流水线：上传完成后后台跑 ASR → OCR → 向量化，状态回写 videos 表
import pipeline
# 配置统一入口：os.environ > 项目内 .env（不再读其他项目的绝对路径）
import settings

# Windows 上 httpx 会读系统代理设置，连 127.0.0.1 的 Qdrant/MinIO 会被代理拦成 502。
# 把本地地址加进 NO_PROXY，保证中间件直连。
_no_proxy = os.environ.get("NO_PROXY", "")
if "127.0.0.1" not in _no_proxy:
    os.environ["NO_PROXY"] = ",".join(filter(None, [_no_proxy, "127.0.0.1", "localhost"]))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

# [面试] session 存 Redis：登录会话放 Redis，而不是默认的签名 cookie——
# 好处：① 服务重启登录态不丢 ② 未来多进程/多机部署时会话共享（这是 Redis 的真实用途，不是"高并发"）
from flask_session import Session
import redis


# ---------- 0. 配置加载（.env -> 环境变量；密钥不硬编码、不进仓库） ----------
settings.load_env()

app = Flask(__name__)
# 密钥从环境变量读，缺失时给一个开发占位（生产必须覆盖）
app.secret_key = os.environ.get("SECRET_KEY", "video-rag-dev-secret")
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_REDIS"] = redis.Redis(
    host=os.environ.get("REDIS_HOST", "127.0.0.1"),
    port=int(os.environ.get("REDIS_PORT", "6379")),
    password=os.environ.get("REDIS_PASSWORD") or None,
)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)

# ---------- 1. 三个客户端 ----------
dashscope_key = os.environ.get("DASHSCOPE_KEY", "")
deepseek_key = os.environ.get("DEEPSEEK_KEY", "")

embed_client = OpenAI(
    api_key=dashscope_key or "placeholder",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60,
    max_retries=3,      # 失败重试：超时/5xx 指数退避
)
chat_client = OpenAI(
    api_key=deepseek_key or "placeholder",
    base_url="https://api.deepseek.com",
    timeout=120,
    max_retries=3,      # 长回答容易超时，同样重试
)
qdrant = QdrantClient(host="127.0.0.1", port=6333)

# [面试] Redis 除会话外，还有两个真实用途：热点缓存 + 限流。
# 复用 app.config 里那个 SESSION_REDIS 连接，不另起连接。
cache = app.config["SESSION_REDIS"]

# [面试] MinIO 对象存储：存用户上传的视频。账号密码从 .env 读，不硬编码
minio_client = Minio(
    os.environ.get("MINIO_ENDPOINT", "127.0.0.1:9000"),
    access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    secure=False,  # 本地 http，不上 https
)
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "videos")

# 本地文件目录：解析流水线要读本地视频（faster-whisper / ffmpeg 都要路径）
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
CHUNK_DIR = os.path.join(UPLOAD_DIR, "_chunks")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 确保 bucket 存在（bucket 相当于"一个放文件的文件夹"）
if not minio_client.bucket_exists(MINIO_BUCKET):
    minio_client.make_bucket(MINIO_BUCKET)

COLLECTION = "video_sentences"


# [面试] 关键词兜底检索：Qdrant 挂了时的降级方案。
# 从 sentences.json / frames_ocr.json 本地按关键词匹配，不依赖向量库。
# 说明"依赖不可用时服务不瘫"——这是工程里"优雅降级"的思路。
def _keyword_search(question, limit=3):
    """Qdrant 不可用时，退化为本地关键词匹配。返回与 search() 相同结构的 hits。"""
    keywords = [w for w in question if w not in "的了吗呢啊吧？?，。、 "]
    base = os.path.dirname(__file__)
    hits = []
    for src, path, label in (("asr", "sentences.json", "语音"),
                             ("ocr", "frames_ocr.json", "画面字幕")):
        fp = os.path.join(base, path)
        if not os.path.exists(fp):
            continue
        for item in json.load(open(fp, encoding="utf-8")):
            text = item.get("text", "")
            # 命中任一关键词即算匹配；给一个粗略的相关性分（命中关键词个数）
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                hits.append({
                    "start": item.get("start", 0),
                    "end": item.get("end", 0),
                    "text": text,
                    "source": src,
                    "score": float(score),
                })
    # 按相关性分降序，取前 limit
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:limit]


def _chunks_for(media_id):
    """按需生成/读取分段摘要。首次会调 LLM（每 5 分钟一次），之后走 SQLite 缓存。"""
    if not media_id:
        return []
    try:
        import chunks as chunks_mod
        return chunks_mod.ensure(int(media_id), media_id=int(media_id))
    except Exception as e:
        print("[chunks] 分段摘要不可用:", type(e).__name__)
        return []


def search(question, limit=3, media_id=None):
    """问题 -> 检索（向量 + BM25 → RRF → 扩窗 → rerank）-> 上下文 + 命中。

    media_id：只搜这个视频的证据（多视频隔离）；不传则全库搜。
    [面试] 和 AgentLoop 走同一套 retrieval.py：两种问答模式如果各用一套检索，
    面试时讲不清，线上也会出现"同一个问题两种模式答案不一样"。
    优雅降级：整条流水线不可用时退化为本地关键词匹配，接口仍能返回结果。
    """
    try:
        from retrieval import search as _retrieval_search
        hits = _retrieval_search(question, limit=limit, media_id=media_id, mode="hybrid+rerank")
        hit_dicts = [{
            "start": h["start"],
            "end": h["end"],
            "text": h["text"],
            "source": h["source"],
            "score": round(h.get("rerank", h.get("rrf", h.get("vec", 0.0))), 4),
        } for h in hits]
    except Exception as e:
        # [面试] 依赖挂了不崩，退化到关键词检索。打印日志便于排查。
        print("[search] 混合检索不可用，降级为关键词检索:", type(e).__name__)
        hit_dicts = _keyword_search(question, limit)

    context = []
    hit_list = []
    for h in hit_dicts:
        sec = h["start"] / 1000
        src = h["source"]
        src_label = "语音" if src == "asr" else "画面字幕"
        context.append("[第 %02d:%02d 秒·%s] %s" % (sec // 60, sec % 60, src_label, h["text"]))
        hit_list.append(h)
    return context, hit_list


# ---------- 1.5 Redis 热点缓存 + 令牌桶限流 ----------
# [面试] 这两个是 Redis 除"会话"外的真实用途，不是"高并发"标签：
#   热点缓存：同一问题重复问，直接返回缓存答案，省 embedding + LLM 的 API 钱
#   令牌桶限流：防止有人狂刷接口烧 API 钱（每个用户一个桶，容量/速率可调）
CACHE_TTL = 3600          # 热点答案缓存 1 小时
RATE_LIMIT_CAPACITY = 10  # 令牌桶容量（突发最多 10 次）
RATE_LIMIT_REFILL = 2     # 每秒补充 2 个令牌


def _cache_get(question):
    """命中热点缓存则返回答案，否则 None。缓存键带前缀避免和 session 冲突。"""
    try:
        return cache.get("qa:" + question)
    except Exception:
        return None  # Redis 挂了不阻断主流程


def _cache_set(question, answer_json):
    try:
        cache.setex("qa:" + question, CACHE_TTL, answer_json)
    except Exception:
        pass  # 缓存失败不影响返回


def _rate_limit_allow(user_key):
    """滑动窗口限流。返回 True 放行 / False 拒绝。

    用 Redis 的 ZSet（有序集合）记录每次请求的时间戳，score=时间：
    每次请求先删掉 1 秒前的时间戳，再数当前窗口内已用的次数，
    超过容量就拒绝。相比令牌桶更直观，且天然原子（pipeline 一次执行）。
    """
    key = "rl:" + user_key
    now = time.time()
    try:
        pipe = cache.pipeline()
        # 删掉 1 秒前的时间戳（滑出窗口的请求）
        pipe.zremrangebyscore(key, 0, now - 1)
        # 数当前 1 秒窗口内已用了几个
        pipe.zcard(key)
        # 记录本次请求的时间戳
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, 2)  # 整个键 2 秒后过期，防 key 堆积
        _, count, _, _ = pipe.execute()
        return count < RATE_LIMIT_CAPACITY
    except Exception:
        return True  # Redis 挂了不阻断（限流是保护措施，不是核心功能）


def answer(question, media_id=None):
    """检索 + 生成，返回 (回答文本, 命中列表)。"""
    context, hit_list = search(question, media_id=media_id)
    system = (
        "你是视频内容问答助手。下面是从视频里检索到的、和用户问题最相关的片段，"
        "每个片段都带时间戳和来源（'语音'是转写的人声，'画面字幕'是识别出的画面文字）。"
        "请只用这些片段回答，并指出答案来自视频的哪个时间点、是语音还是画面字幕。"
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
    return resp.choices[0].message.content, hit_list


# ---------- 3. 接口 ----------
# [面试] 生产部署：Flask 直接 serve Vue 构建产物（client/dist）。
# 开发时前端跑 Vite dev server（5173）通过 /api 代理到本后端，互不干扰。
DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client", "dist")


@app.route("/")
def index():
    # 有 Vue 构建产物就返回它，否则回退到 templates 里的旧版页面
    if os.path.exists(os.path.join(DIST_DIR, "index.html")):
        return open(os.path.join(DIST_DIR, "index.html"), encoding="utf-8").read()
    return render_template("index.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    """serve Vue 构建产物里的静态资源（js/css）。"""
    from flask import send_from_directory
    return send_from_directory(os.path.join(DIST_DIR, "assets"), filename)


# [面试] AgentLoop 版问答：走 Planner -> Executor -> Critic 闭环（最多2轮），
# 区别于 /ask 的单轮 RAG。返回结构化产物 + 审核结果。
@app.route("/api/ask_agent", methods=["POST"])
def ask_agent():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    media_id = data.get("mediaId")
    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    # [面试] 多轮追问改写（和 /api/ask 同一套）
    from query_rewrite import rewrite
    search_q, rewritten = rewrite(question, data.get("history") or [])

    # [面试] 限流 + 热点缓存：Agent 链路比单轮 RAG 更烧钱（Planner+Executor+Critic 多轮 LLM），
    # 更需要防刷和缓存兜底。缓存键带 mediaId + 改写后的问题。
    user_key = str(session.get("user_id") or request.remote_addr or "anon")
    if not _rate_limit_allow(user_key):
        return jsonify({"error": "请求太频繁，请稍后再试"}), 429
    cache_key = "agent:m%s:%s" % (media_id, search_q)
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(json.loads(cached.decode("utf-8") if isinstance(cached, bytes) else cached))

    t0 = time.time()
    try:
        result = run_agent(search_q, media_id=media_id, chunks=_chunks_for(media_id))
    except Exception as e:
        return jsonify({"error": type(e).__name__, "detail": str(e)}), 500
    if rewritten:
        result["rewritten"] = search_q
    cost_ms = int((time.time() - t0) * 1000)

    # [面试] 异步落库：记进 SQLite/MySQL（任务记录 + 评测指标），
    # 走线程队列不阻塞返回；带 user_id 做多用户隔离（未登录则 None=匿名）
    storage.record_task(
        question=question,
        result=result.get("result"),
        metrics={
            "criticPassed": bool(result.get("critique", {}).get("passed")),
            "rounds": result.get("rounds"),
            "evidencePassed": (result.get("evidence_check") or {}).get("passed"),
            "evidenceTotal": (result.get("evidence_check") or {}).get("total"),
        },
        answer_mode="agent",
        rounds=result.get("rounds", 1),
        cost_ms=cost_ms,
        user_id=session.get("user_id"),
    )
    # 缓存 Agent 结果（同 question 命中直接返回，省多轮 LLM 的钱）
    _cache_set(cache_key, json.dumps(result, ensure_ascii=False))
    return jsonify(result)


# [面试] SSE 流式问答：把 AgentLoop 每个阶段（检索/Planner/Executor/Critic）
# 实时推给前端，前端展示「Agent 正在做什么」——对应原项目 DOVideo-AI 的实时进度推送。
# 核心：run_agent 加 on_event 回调，这里把回调事件转成 SSE 的 data 行。
@app.route("/api/ask_agent_stream", methods=["POST"])
def ask_agent_stream():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    media_id = data.get("mediaId")
    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    user_key = str(session.get("user_id") or request.remote_addr or "anon")
    if not _rate_limit_allow(user_key):
        return jsonify({"error": "请求太频繁，请稍后再试"}), 429

    # [面试] 多轮追问改写：流开始前算好，前端能先看到"改写成什么"
    from query_rewrite import rewrite
    search_q, rewritten = rewrite(question, data.get("history") or [])

    import queue as _queue
    from agent_loop import run_agent as _run_agent

    def generate():
        # 用线程队列把 run_agent 的回调事件接出来，逐条写成 SSE 帧
        q = _queue.Queue()

        # 先推一条改写事件（没改写就不推），前端点亮「问题改写」阶段
        if rewritten:
            yield "data: %s\n\n" % json.dumps(
                {"stage": "rewrite", "status": "done", "query": search_q}, ensure_ascii=False)

        def on_event(ev):
            q.put(ev)

        result_holder = {}

        def _work():
            try:
                result_holder["result"] = _run_agent(
                    search_q, on_event=on_event, media_id=media_id,
                    chunks=_chunks_for(media_id),
                )
            except Exception as e:
                result_holder["error"] = {"type": type(e).__name__, "detail": str(e)}
            finally:
                q.put(None)  # 结束哨兵

        import threading
        t = threading.Thread(target=_work, daemon=True)
        t.start()

        while True:
            ev = q.get()
            if ev is None:
                break
            # 每个事件写成一条 SSE 帧：data: {json}\n\n
            yield "data: %s\n\n" % json.dumps(ev, ensure_ascii=False)

        if "error" in result_holder:
            yield "data: %s\n\n" % json.dumps(
                {"stage": "error", "error": result_holder["error"]}, ensure_ascii=False)
            return

        # 正常结束：发最终产物帧
        result = result_holder["result"]
        t0 = time.time()
        storage.record_task(
            question=question,
            result=result.get("result"),
            metrics={
                "criticPassed": bool(result.get("critique", {}).get("passed")),
                "rounds": result.get("rounds"),
                "evidencePassed": (result.get("evidence_check") or {}).get("passed"),
                "evidenceTotal": (result.get("evidence_check") or {}).get("total"),
            },
            answer_mode="agent",
            rounds=result.get("rounds", 1),
            cost_ms=0,
            user_id=session.get("user_id"),
        )
        yield "data: %s\n\n" % json.dumps(
            {"stage": "done", "result": result}, ensure_ascii=False)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关 nginx 缓冲，保证实时（部署时有用）
        },
    )


# [面试] 反馈接口：用户点"满意/不满意"，存进 feedback 表，算 userAcceptanceRate 的数据源
@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    task_id = data.get("task_id")
    rating = data.get("rating")  # 1=满意，-1=不满意
    comment = data.get("comment", "")
    if rating not in (1, -1):
        return jsonify({"error": "rating 必须是 1(满意) 或 -1(不满意)"}), 400
    storage.record_feedback(task_id, rating, comment)
    return jsonify({"ok": True, "rating": rating})


@app.route("/api/stats", methods=["GET"])
def stats():
    """统计接口：返回最近任务 + 用户接受率，前端/面试演示用（多用户时按登录用户过滤）。"""
    user_id = session.get("user_id")
    return jsonify({
        "loggedIn": user_id is not None,
        "userId": user_id,
        "recentTasks": storage.query_recent_tasks(20, user_id=user_id),
        "userAcceptanceRate": storage.user_acceptance_rate(),
    })


# ============ 多用户登录体系 ============
# [面试] 注册/登录/登出：session 存 Redis，用户隔离是"多用户产品"的前提。
# 密码只存哈希（storage._hash_password），不存明文——安全基础，面试必讲。

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少 6 位"}), 400
    ok, msg = storage.create_user(username, password)
    if not ok:
        # 重名等冲突，msg 已经是友好中文提示（如"用户名已存在"）
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True, "message": "注册成功，请登录"})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    user_id = storage.verify_user(username, password)
    if user_id is None:
        return jsonify({"error": "用户名或密码错误"}), 401
    # 登录成功：把 user_id 写进 session（存 Redis），后续接口靠它做用户隔离
    session["user_id"] = user_id
    session["username"] = username
    return jsonify({"ok": True, "userId": user_id, "username": username})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()  # 清 session = 登出
    return jsonify({"ok": True})


@app.route("/api/me", methods=["GET"])
def me():
    """当前登录用户信息（前端用来判断是否已登录、显示用户名）。"""
    if "user_id" in session:
        return jsonify({"loggedIn": True, "userId": session["user_id"], "username": session.get("username")})
    return jsonify({"loggedIn": False})


@app.route("/api/ask", methods=["POST"])
def ask():
    # [面试] 前端可能没传 question，或传了非字符串，都要兜住，别让服务崩
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    media_id = data.get("mediaId")
    history = data.get("history") or []
    if not question:
        return jsonify({"error": "question 不能为空"}), 400

    # [面试] 多轮追问改写：把「那第二部呢」补成「本季高分作的第二部是什么题材」
    from query_rewrite import rewrite
    search_q, rewritten = rewrite(question, history)

    # [面试] 限流：匿名用户用 IP 作 key，登录用户用 user_id，防刷烧 API
    user_key = str(session.get("user_id") or request.remote_addr or "anon")
    if not _rate_limit_allow(user_key):
        return jsonify({"error": "请求太频繁，请稍后再试"}), 429

    # [面试] 热点缓存：同一问题命中缓存直接返回，省 embedding + LLM 的钱。
    # 缓存键要带 mediaId 和改写后的问题，否则 A 视频 / 另一种问法会复用错答案。
    cache_key = "m%s:%s" % (media_id, search_q)
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(json.loads(cached.decode("utf-8") if isinstance(cached, bytes) else cached))

    try:
        ans, hits = answer(search_q, media_id=media_id)
    except Exception as e:
        # [面试] 兜住异常，返回可读错误，不把堆栈泄露给前端
        return jsonify({"error": type(e).__name__, "detail": str(e)}), 500

    result = {"answer": ans, "hits": hits}
    if rewritten:
        result["rewritten"] = search_q
    _cache_set(cache_key, json.dumps(result, ensure_ascii=False))
    return jsonify(result)


# ------------------------------------------------------------
# 上传与自动解析
# ------------------------------------------------------------
ALLOWED_EXT = ("mp4", "mkv", "mov", "avi", "wav", "mp3", "m4a", "flv")


def _ingest_video(user_id, filename, data):
    """保存视频（本地 + MinIO）→ 记元数据 → 触发后台解析。返回 (video_id, dedup)。

    为什么要留一份本地文件：faster-whisper 和 ffmpeg 都要读本地路径，
    从 MinIO 现拉会多一次网络拷贝。解析中间产物落 uploads/<video_id>/。
    """
    import hashlib

    content_hash = hashlib.sha256(data).hexdigest()
    existing = storage.query_video_by_hash(content_hash)
    if existing:
        # 同一内容已解析过，直接复用；返回已有对象名，前端可立即播放
        return existing["id"], existing["object_name"], True

    local_name = "%s_%s" % (uuid.uuid4().hex[:8], filename)
    local_path = os.path.join(UPLOAD_DIR, local_name)
    with open(local_path, "wb") as f:
        f.write(data)

    minio_client.put_object(
        MINIO_BUCKET, local_name, io.BytesIO(data),
        length=len(data), content_type="application/octet-stream",
    )
    vid = storage.record_video_sync(user_id or 0, filename, local_name, len(data), content_hash)
    pipeline.enqueue(vid, local_path)        # 后台跑 ASR → OCR → 向量化
    return vid, local_name, False


# [面试] 上传接口：接收视频文件，存进 MinIO。这是"上传"和"提交文字"的本质区别——
# 文字用 JSON（application/json），文件用 FormData（multipart/form-data），后端用 request.files 接
@app.route("/api/upload", methods=["POST"])
def upload():
    """单文件上传（小视频）。大文件走下面的分片接口。"""
    if "file" not in request.files:
        return jsonify({"error": "没有收到文件"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    filename = f.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "不支持的文件格式: " + ext}), 400

    vid, obj, dedup = _ingest_video(session.get("user_id"), filename, f.read())
    return jsonify({
        "ok": True,
        "videoId": vid,
        "objectName": obj,
        "deduplicated": dedup,
        "filename": filename,
        "message": "视频已存在，复用已有索引" if dedup else "上传成功，已开始后台解析",
    })


# [面试] 分片上传 + 断点续传：大文件切成 5MB 一片，Redis/磁盘记已传分片，
# 弱网中断后前端先查 /status，只补传缺的那几片。
@app.route("/api/upload/init", methods=["POST"])
def upload_init():
    """分片上传第一步：登记文件名，返回 uploadId。"""
    data = request.get_json(silent=True) or {}
    filename = (data.get("filename") or "").strip()
    if not filename:
        return jsonify({"error": "filename 不能为空"}), 400
    upload_id = uuid.uuid4().hex[:16]
    d = os.path.join(CHUNK_DIR, upload_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"filename": filename}, f, ensure_ascii=False)
    return jsonify({"ok": True, "uploadId": upload_id})


@app.route("/api/upload/chunk", methods=["POST"])
def upload_chunk():
    """上传一个分片。index 从 0 开始。"""
    upload_id = request.form.get("uploadId", "")
    index = request.form.get("index", "")
    f = request.files.get("chunk")
    if not upload_id or not index.isdigit() or not f:
        return jsonify({"error": "参数不完整"}), 400
    d = os.path.join(CHUNK_DIR, upload_id)
    if not os.path.isdir(d):
        return jsonify({"error": "uploadId 不存在"}), 404
    f.save(os.path.join(d, "%06d.part" % int(index)))
    return jsonify({"ok": True, "index": int(index)})


@app.route("/api/upload/status", methods=["GET"])
def upload_status():
    """断点续传：告诉前端哪些分片已经收到了。"""
    upload_id = request.args.get("uploadId", "")
    d = os.path.join(CHUNK_DIR, upload_id)
    if not os.path.isdir(d):
        return jsonify({"error": "uploadId 不存在"}), 404
    done = sorted(int(n.split(".")[0]) for n in os.listdir(d) if n.endswith(".part"))
    return jsonify({"ok": True, "uploaded": done})


@app.route("/api/upload/complete", methods=["POST"])
def upload_complete():
    """所有分片到齐后合并 → 入库 → 触发解析。"""
    data = request.get_json(silent=True) or {}
    upload_id = data.get("uploadId") or ""
    d = os.path.join(CHUNK_DIR, upload_id)
    if not os.path.isdir(d):
        return jsonify({"error": "uploadId 不存在"}), 404

    meta_path = os.path.join(d, "_meta.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "缺少上传元数据"}), 400
    filename = json.load(open(meta_path, encoding="utf-8"))["filename"]

    parts = sorted(n for n in os.listdir(d) if n.endswith(".part"))
    if not parts:
        return jsonify({"error": "没有收到任何分片"}), 400

    buf = bytearray()
    for p in parts:
        with open(os.path.join(d, p), "rb") as f:
            buf.extend(f.read())

    vid, obj, dedup = _ingest_video(session.get("user_id"), filename, bytes(buf))
    # 合并成功后再清分片，避免"清完才发现入库失败"
    for p in parts:
        try:
            os.remove(os.path.join(d, p))
        except OSError:
            pass
    return jsonify({
        "ok": True,
        "videoId": vid,
        "objectName": obj,
        "deduplicated": dedup,
        "filename": filename,
        "message": "视频已存在，复用已有索引" if dedup else "合并完成，已开始后台解析",
    })


@app.route("/api/video-status", methods=["GET"])
def video_status():
    """轮询解析进度：前端上传完调这个看 status/progress。"""
    vid = request.args.get("id", type=int)
    if not vid:
        return jsonify({"error": "id 不能为空"}), 400
    v = storage.query_video(vid)
    if not v:
        return jsonify({"error": "视频不存在"}), 404
    return jsonify({"video": v})


@app.route("/api/playback")
def playback():
    """从 MinIO 流式返回视频，支持 Range（<video> 拖进度条、跳帧都要用）。"""
    name = request.args.get("name", "")
    if not name:
        return jsonify({"error": "name 不能为空"}), 400
    try:
        stat = minio_client.stat_object(MINIO_BUCKET, name)
    except Exception:
        return jsonify({"error": "视频不存在"}), 404

    size = stat.size
    start, end = 0, size - 1
    status = 200
    rng = request.headers.get("Range", "")
    if rng.startswith("bytes="):
        spec = rng[6:].split(",")[0]
        s, _, e = spec.partition("-")
        if s.isdigit():
            start = int(s)
        if e.isdigit():
            end = min(int(e), size - 1)
        if start > end or start >= size:
            return Response(status=416)
        status = 206

    length = end - start + 1
    obj = minio_client.get_object(MINIO_BUCKET, name, offset=start, length=length)

    def gen():
        try:
            for chunk in obj.stream(64 * 1024):
                yield chunk
        finally:
            try:
                obj.close()
                obj.release_conn()
            except Exception:
                pass

    resp = Response(stream_with_context(gen()), status=status, mimetype="video/mp4")
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(length)
    if status == 206:
        resp.headers["Content-Range"] = "bytes %d-%d/%d" % (start, end, size)
    return resp


# ------------------------------------------------------------
# 健康检查
# ------------------------------------------------------------
@app.route("/health")
def health():
    """进程活着 + 依赖状态。依赖挂了不代表进程该死，所以总返回 200。"""
    deps = {}
    try:
        cache.ping()
        deps["redis"] = "up"
    except Exception:
        deps["redis"] = "down"
    try:
        minio_client.bucket_exists(MINIO_BUCKET)
        deps["minio"] = "up"
    except Exception:
        deps["minio"] = "down"
    try:
        qdrant.get_collections()
        deps["qdrant"] = "up"
    except Exception:
        deps["qdrant"] = "down"

    # 缺 key 时进程没死，但检索会静默退化成 BM25，所以要单独报出来（不是"健康"）
    keys = settings.key_status()
    healthy = all(v == "up" for v in deps.values()) and all(keys.values())
    return jsonify({
        "status": "up" if healthy else "degraded",
        "deps": deps,
        "keys": keys,
    })


# [面试] 视频列表接口：工作台左侧显示"该用户上传的视频"，多用户隔离
@app.route("/api/videos", methods=["GET"])
def videos():
    user_id = session.get("user_id")
    return jsonify({"videos": storage.query_videos(user_id=user_id)})


if __name__ == "__main__":
    # [面试] 生产环境不用 debug=True（会暴露调试信息），本地开发才开
    app.run(host="127.0.0.1", port=8001, debug=False)
