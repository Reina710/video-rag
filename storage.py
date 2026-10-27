# -*- coding: utf-8 -*-
"""
存储抽象层：一个模块，两套后端（SQLite 本地开发 / MySQL 生产），
上层调用方不关心底层是哪个，只调统一的函数。

为什么做"抽象层"而不是直接写死 MySQL：
 问"为什么用 MySQL"时，正确回答是"多人多机场景下 SQLite 是单文件不共享，
  生产必须换 MySQL"。抽象层的价值：证明你知道存储是可以替换的，
  本地开发图省事用 SQLite，生产改一个环境变量就切 MySQL，SQL 层完全一样。

表结构：
  1. users        用户表（多用户登录的基础，业务表都带 user_id 做隔离）
  2. videos       视频元数据表（每个用户上传的视频，含归属 user_id）
  3. task_records 任务记录（每次问答/评测，含归属 user_id）
  4. feedback     用户反馈（rating 1/-1，算 userAcceptanceRate）

异步落库：
  写库 IO 不能挂在响应链路上，否则一次慢查询就把接口拖住。
  这里用 queue.Queue + 后台线程实现"接口先返回、落库后台做"；
  单机部署够用，上多机会换成消息队列，接口这边的语义不变。

选择后端：环境变量 STORAGE=mysql 则用 MySQL，否则默认 SQLite。
"""
import sqlite3
import threading
import queue
import json
import time
import os
import hashlib


def _load_env(path=".env"):
    """极简 .env 加载（与 app.py 保持一致）：密钥从环境变量读，不硬编码、不进仓库。"""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_env(os.path.join(os.path.dirname(__file__), ".env"))

# ---------- 0. 后端选择 ----------
STORAGE_BACKEND = os.environ.get("STORAGE", "sqlite").lower()  # "sqlite" / "mysql"

if STORAGE_BACKEND == "mysql":
    import pymysql
    MYSQL_CONFIG = {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "user": os.environ.get("MYSQL_USER", "root"),
        "password": os.environ.get("MYSQL_PASSWORD", ""),
        "database": os.environ.get("MYSQL_DATABASE", "video_rag"),
        "charset": "utf8mb4",
    }
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), "video_rag.db")


# ---------- 1. 连接管理 ----------
def _get_conn():
    """按后端返回一个新连接。每个调用方/线程各自建连接，不跨线程共享。"""
    if STORAGE_BACKEND == "mysql":
        return pymysql.connect(**MYSQL_CONFIG)
    return sqlite3.connect(DB_PATH)


def _placeholder(n):
    """SQLite 用 ?，MySQL(pymysql) 也认 %s 但统一用 ? 会被 pymysql 转。这里用 %s 兼容 MySQL。"""
    if STORAGE_BACKEND == "mysql":
        return ", ".join(["%s"] * n)
    return ", ".join(["?"] * n)


def _autoinc():
    """自增主键语法。SQLite 用 AUTOINCREMENT，MySQL 用 AUTO_INCREMENT。"""
    return "AUTO_INCREMENT" if STORAGE_BACKEND == "mysql" else "AUTOINCREMENT"


# ---------- 2. 建表 ----------
_schema_ready = False   # 本进程已建过表就不再跑（MySQL 下每次省 8 次网络往返）


def init_db():
    """建表（幂等）。首次调用执行，之后直接返回。

    每个读写函数入口都会调它，所以必须廉价：SQLite 是本地文件无所谓，
    MySQL 每次要 6 条 CREATE TABLE IF NOT EXISTS + 2 条 ALTER，
    挂在问答路径上就是 8 次 round-trip。
    """
    global _schema_ready
    if _schema_ready:
        return
    conn = _get_conn()
    cur = conn.cursor()
    ai = _autoinc()

    # 用户表：多用户登录的基础
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY %s,
            username     VARCHAR(64) NOT NULL UNIQUE,
            password_hash VARCHAR(128) NOT NULL,          -- 只存哈希，不存明文（安全基础）
            created_at   TEXT NOT NULL
        )
    """ % ai)

    # 视频元数据表：每个视频归属一个用户（多用户隔离的核心）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id           INTEGER PRIMARY KEY %s,
            user_id      INTEGER NOT NULL,                 -- 归属用户
            filename     VARCHAR(255) NOT NULL,
            object_name  VARCHAR(255) NOT NULL,            -- MinIO 里的对象名
            size_bytes   INTEGER,
            content_hash VARCHAR(64),                       -- 文件内容哈希（SHA256），用于上传去重
            status       VARCHAR(32) DEFAULT 'uploaded',    -- uploaded/processing/done/failed
            created_at   TEXT NOT NULL
        )
    """ % ai)

    # 任务记录表（加 user_id，多用户隔离）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS task_records (
            id           INTEGER PRIMARY KEY %s,
            user_id      INTEGER,                           -- 归属用户（可能为 None=匿名）
            question     TEXT NOT NULL,
            answer_mode  VARCHAR(16) NOT NULL DEFAULT 'agent',
            result       TEXT,
            metrics      TEXT,
            rounds       INTEGER DEFAULT 1,
            cost_ms      INTEGER,
            created_at   TEXT NOT NULL
        )
    """ % ai)

    # 反馈表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id           INTEGER PRIMARY KEY %s,
            task_id      INTEGER,
            rating       INTEGER NOT NULL,
            comment      TEXT,
            created_at   TEXT NOT NULL
        )
    """ % ai)

    # 分段摘要表：长视频按 5 分钟切块，每块一段摘要。
    # 解决"时间坍缩"——光靠检索约束治标，分段目录才能让模型看到全片。
    cur.execute("""
        CREATE TABLE IF NOT EXISTS video_chunks (
            id         INTEGER PRIMARY KEY %s,
            video_id   INTEGER NOT NULL,
            start_ms   INTEGER NOT NULL,
            end_ms     INTEGER NOT NULL,
            summary    TEXT,
            keywords   TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(video_id, start_ms)
        )
    """ % ai)

    conn.commit()
    conn.close()

    # 轻量迁移：旧库的 videos 表没有 content_hash 列时补上（幂等，新库会忽略报错）
    _migrate_add_content_hash()
    _migrate_add_progress()

    _schema_ready = True


def _migrate_add_progress():
    """旧库补 progress 列（解析进度文案，如"转写中"）。失败静默（列已存在）。"""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("ALTER TABLE videos ADD COLUMN progress VARCHAR(64)")
        conn.commit()
        conn.close()
    except Exception:
        pass


def _migrate_add_content_hash():
    """旧库补 content_hash 列。SQLite/MySQL 都试一遍，失败静默（列已存在）。"""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        if STORAGE_BACKEND == "mysql":
            cur.execute("ALTER TABLE videos ADD COLUMN content_hash VARCHAR(64)")
        else:
            cur.execute("ALTER TABLE videos ADD COLUMN content_hash VARCHAR(64)")
        conn.commit()
        conn.close()
    except Exception:
        pass  # 列已存在或其它，忽略


# ---------- 3. 用户（同步，登录/注册需要即时结果） ----------
def _hash_password(password):
    """密码哈希（加盐）。原则：绝不存明文密码，存 hash，撞库也拿不到原密码。"""
    salt = os.environ.get("PASSWORD_SALT", "video-rag-salt")  # 生产用随机盐 + bcrypt
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def create_user(username, password):
    """注册用户。返回 (ok, msg)。重名返回 False。"""
    init_db()
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (%s)"
            % _placeholder(3),
            (username, _hash_password(password), time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        return True, "注册成功"
    except Exception as e:
        # 唯一约束冲突（重名）等，返回友好提示而不是英文 SQL 错误
        if "UNIQUE" in str(e).upper() or "duplicate" in str(e).lower():
            return False, "用户名已存在"
        return False, str(e)
    finally:
        conn.close()


def verify_user(username, password):
    """校验用户名密码。返回 user_id 或 None。"""
    init_db()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, password_hash FROM users WHERE username = %s" % _placeholder(1),
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    user_id, stored_hash = row
    if stored_hash == _hash_password(password):
        return user_id
    return None


# ---------- 4. 异步写队列（等价于消息队列削峰） ----------
_write_queue = queue.Queue()
_worker_started = False


def _is_disconnect(e):
    """判断是不是"连接已断"（而不是 SQL 写错）。

    MySQL 会按 wait_timeout 主动踢掉空闲连接：后台线程一直握着一个连接不动，
    几小时后第一次写就报 2006 gone away / 2013 lost connection。
    这种情况重连一次就能继续，不重连会让整个进程后续写库全部失败。
    """
    code = e.args[0] if getattr(e, "args", None) else None
    return code in (2006, 2013)


def _worker_loop():
    """后台线程：从队列取写库任务，逐条落库。

    连接断了就重建再试一次；真写不进去才放弃这一条（并打日志，不再静默吞掉）。
    """
    conn = _get_conn()
    while True:
        item = _write_queue.get()
        if item is None:
            break
        table, data = item
        try:
            _insert(conn, table, data)
            conn.commit()
        except Exception as e:
            if _is_disconnect(e):
                try:
                    conn.close()
                except Exception:
                    pass
                conn = _get_conn()          # 换一条新连接，这一条重试
                try:
                    _insert(conn, table, data)
                    conn.commit()
                    continue
                except Exception as e2:
                    e = e2
            print("[storage] 异步落库失败:", table, type(e).__name__, e, flush=True)
            try:
                conn.rollback()
            except Exception:
                pass
    conn.close()


def _insert(conn, table, data):
    """按表名插入一条。data 是 dict，键=列名。用 cursor 兼容 sqlite3 和 pymysql。"""
    cols = ", ".join(data.keys())
    ph = _placeholder(len(data))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO %s (%s) VALUES (%s)" % (table, cols, ph),
        tuple(data.values()),
    )
    cur.close()


def _ensure_worker():
    global _worker_started
    if not _worker_started:
        _worker_started = True
        init_db()
        t = threading.Thread(target=_worker_loop, daemon=True)
        t.start()


def record_task(question, result, metrics=None, answer_mode="agent", rounds=1, cost_ms=None, user_id=None):
    """异步记录一次任务（入队即返回）。"""
    _ensure_worker()
    _write_queue.put((
        "task_records",
        {
            "question": question,
            "answer_mode": answer_mode,
            "result": json.dumps(result, ensure_ascii=False) if result else None,
            "metrics": json.dumps(metrics, ensure_ascii=False) if metrics else None,
            "rounds": rounds,
            "cost_ms": cost_ms,
            "user_id": user_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    ))


def record_feedback(task_id, rating, comment=""):
    """异步记录一条用户反馈。"""
    _ensure_worker()
    _write_queue.put((
        "feedback",
        {
            "task_id": task_id,
            "rating": rating,
            "comment": comment,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    ))


def record_video(user_id, filename, object_name, size_bytes, content_hash=None):
    """异步记录一条视频元数据。content_hash 是文件 SHA256，用于上传去重。"""
    _ensure_worker()
    _write_queue.put((
        "videos",
        {
            "user_id": user_id,
            "filename": filename,
            "object_name": object_name,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "status": "uploaded",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    ))


def record_video_sync(user_id, filename, object_name, size_bytes, content_hash=None):
    """同步插入视频元数据，返回 video_id。

    上传接口拿到 id 才能立刻触发后台解析（异步队列只负责"写库"，不能负责"要 id"）。
    """
    init_db()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO videos (user_id, filename, object_name, size_bytes, content_hash, status, created_at) "
        "VALUES (%s)" % _placeholder(7),
        (user_id, filename, object_name, size_bytes, content_hash,
         "uploaded", time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    vid = cur.lastrowid
    conn.close()
    return vid


def update_video(video_id, status=None, progress=None):
    """更新视频的解析状态 / 进度。两个参数传哪个改哪个。"""
    if status is None and progress is None:
        return
    sets = []
    params = []
    if status is not None:
        sets.append("status = %s" % _placeholder(1))
        params.append(status)
    if progress is not None:
        sets.append("progress = %s" % _placeholder(1))
        params.append(progress)
    params.append(video_id)
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE videos SET %s WHERE id = %s" % (", ".join(sets), _placeholder(1)),
        tuple(params),
    )
    conn.commit()
    conn.close()


def query_video(video_id):
    """按 id 查一条视频（含 status/progress），解析进度轮询用。"""
    rows = _fetchall(
        "SELECT id, user_id, filename, object_name, size_bytes, content_hash, status, progress, created_at "
        "FROM videos WHERE id = %s" % _placeholder(1),
        (video_id,),
    )
    return rows[0] if rows else None


def query_video_by_object_name(object_name):
    """按 MinIO 对象名反查视频，用于播放接口的归属校验。

    同名对象可能被多个用户上传（去重前），取 id 最小的那一条；
    真正的权限判断在调用方，这里只负责"这个名字对应哪条记录"。
    """
    rows = _fetchall(
        "SELECT id, user_id, filename, object_name, size_bytes, status "
        "FROM videos WHERE object_name = %s ORDER BY id ASC" % _placeholder(1),
        (object_name,),
    )
    return rows[0] if rows else None


# ---------- 5. 查询（同步读） ----------
def _fetchall(sql, params=()):
    init_db()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


# ---------- 分段摘要 ----------
def save_chunks(video_id, chunks):
    """写入分段摘要（先删后插，幂等）。"""
    init_db()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM video_chunks WHERE video_id = %s" % _placeholder(1), (video_id,))
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for c in chunks:
            cur.execute(
                "INSERT INTO video_chunks (video_id, start_ms, end_ms, summary, keywords, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)" % ((_placeholder(1),) * 6),
                (int(video_id), int(c["startMs"]), int(c["endMs"]), c.get("summary") or "",
                 json.dumps(c.get("keywords") or [], ensure_ascii=False), now),
            )
        conn.commit()
    finally:
        conn.close()


def load_chunks(video_id):
    """读某视频的分段摘要，按时间排序。"""
    rows = _fetchall(
        "SELECT start_ms, end_ms, summary, keywords FROM video_chunks "
        "WHERE video_id = %s ORDER BY start_ms" % _placeholder(1),
        (video_id,),
    )
    out = []
    for r in rows:
        try:
            kws = json.loads(r["keywords"] or "[]")
        except (ValueError, TypeError):
            kws = []
        out.append({
            "startMs": r["start_ms"],
            "endMs": r["end_ms"],
            "summary": r["summary"] or "",
            "keywords": kws,
        })
    return out


_STATUS_RANK = {"done": 3, "processing": 2, "uploaded": 1, "failed": 0}


def _dedup_videos(rows):
    """同一文件名只留一条：优先已解析(done)，其次解析中，最后按 id 大的。

    用户重复上传同一个视频会留下多条记录，列表里看着像"两个一样的历史记录"。
    这里按文件名去重，保留最能用的那条。
    """
    best = {}
    for r in rows:
        name = r["filename"]
        cur = best.get(name)
        if cur is None:
            best[name] = r
            continue
        rk = (_STATUS_RANK.get(r.get("status"), 0), r["id"])
        ck = (_STATUS_RANK.get(cur.get("status"), 0), cur["id"])
        if rk > ck:
            best[name] = r
    return sorted(best.values(), key=lambda r: -r["id"])


def query_videos(user_id=None, limit=50):
    """读视频列表（按用户过滤 + 按文件名去重）。"""
    cols = "id, user_id, filename, object_name, size_bytes, status, progress, created_at"
    if user_id is not None:
        sql = "SELECT %s FROM videos WHERE user_id = %s ORDER BY id DESC" % (cols, _placeholder(1))
        rows = _fetchall(sql, (user_id,))
    else:
        sql = "SELECT %s FROM videos ORDER BY id DESC" % cols
        rows = _fetchall(sql)
    return _dedup_videos(rows)[:limit]


def query_video_by_hash(content_hash):
    """按内容哈希查是否已上传过同一视频（上传去重）。返回命中行或 None。"""
    if not content_hash:
        return None
    rows = _fetchall(
        "SELECT id, user_id, filename, object_name, size_bytes, content_hash, status "
        "FROM videos WHERE content_hash = %s LIMIT 1" % _placeholder(1),
        (content_hash,),
    )
    return rows[0] if rows else None


def query_task(task_id):
    """按 id 查一条任务（反馈时校验归属用）。id 不合法直接当不存在。"""
    try:
        tid = int(task_id)
    except (TypeError, ValueError):
        return None
    rows = _fetchall(
        "SELECT id, user_id FROM task_records WHERE id = %s" % _placeholder(1),
        (tid,),
    )
    return rows[0] if rows else None


def query_recent_tasks(limit=20, user_id=None):
    """读最近 N 条任务记录（可按用户过滤，多用户隔离）。"""
    if user_id is not None:
        sql = "SELECT id, user_id, question, answer_mode, rounds, cost_ms, created_at " \
              "FROM task_records WHERE user_id = %s ORDER BY id DESC LIMIT %s" % (_placeholder(1), _placeholder(1))
        return _fetchall(sql, (user_id, limit))
    sql = "SELECT id, user_id, question, answer_mode, rounds, cost_ms, created_at " \
          "FROM task_records ORDER BY id DESC LIMIT %s" % _placeholder(1)
    return _fetchall(sql, (limit,))


def user_acceptance_rate():
    """算用户接受率（rating>0 占比）。"""
    rows = _fetchall("SELECT rating FROM feedback")
    if not rows:
        return 0.0
    accepted = sum(1 for r in rows if r["rating"] > 0)
    return accepted / len(rows)


# ---------- 6. 自测 ----------
if __name__ == "__main__":
    init_db()
    print("后端:", STORAGE_BACKEND.upper())
    print("后端:", "MySQL" if STORAGE_BACKEND == "mysql" else "SQLite")

    # 测用户
    ok, msg = create_user("test_user", "123456")
    print("注册 test_user:", ok, msg)
    uid = verify_user("test_user", "123456")
    print("登录校验 uid:", uid)
    print("错误密码校验:", verify_user("test_user", "wrong"))

    # 测视频 + 任务 + 反馈
    record_video(uid, "demo.mp4", "demo.mp4", 12345)
    record_task("测试问题", {"title": "测试"}, metrics={"structuredValid": True}, user_id=uid)
    record_feedback(None, 1, "自测")
    time.sleep(1)

    print("最近任务(该用户):", query_recent_tasks(5, user_id=uid))
    print("用户接受率:", user_acceptance_rate())
