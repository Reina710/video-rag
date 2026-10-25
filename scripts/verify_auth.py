"""接口鉴权回归：未登录 / 越权 / 本人 / 非法参数 四类，实测状态码。

用法（服务起来后，在项目根目录执行）：
    python scripts/verify_auth.py
    python scripts/verify_auth.py --base http://127.0.0.1:8001 --media-id 5

脚本会注册一个临时账号跑完矩阵，结束时删掉自己和它产生的数据。
默认 media_id=5 指的是「不属于临时账号」的一条视频，可以用 --media-id 指定。
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar

BASE = "http://127.0.0.1:8001"
MEDIA_ID = 5
ROWS = []


def session():
    """带 cookie 的 opener；每个身份一个，模拟不同浏览器。"""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(CookieJar()),
    )


def call(opener, method, path, payload=None, form=None, headers=None):
    body = None
    hdrs = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode()
        hdrs["Content-Type"] = "application/json"
    if form is not None:
        # 分片上传是 multipart：必须带一个 chunk 文件，否则会先撞上「参数不完整」的 400，
        # 测不到后面的 uploadId 归属校验。
        boundary = "----probe"
        parts = []
        for k, v in form.items():
            parts.append("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                         % (boundary, k, v))
        parts.append(
            "--%s\r\nContent-Disposition: form-data; name=\"chunk\"; filename=\"0.part\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\nprobe-bytes\r\n" % boundary)
        body = ("".join(parts) + "--%s--\r\n" % boundary).encode()
        hdrs["Content-Type"] = "multipart/form-data; boundary=" + boundary
    req = urllib.request.Request(BASE + path, data=body, headers=hdrs, method=method)
    try:
        resp = opener.open(req, timeout=25)
        return resp.status, resp.headers.get("Content-Type", ""), resp.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read(400).decode("utf-8", "replace")
    except Exception as e:  # 连接失败也算不通过
        return "ERR:" + type(e).__name__, "", ""


def case(name, expect, method, path, opener, **kw):
    status, ctype, _body = call(opener, method, path, **kw)
    ok = status == expect
    # SSE 接口的 401 必须走普通 JSON，不能被包进事件流
    if "ask_agent_stream" in path and status == 401 and "event-stream" in (ctype or ""):
        ok = False
    ROWS.append((ok, name, expect, status))
    print("%s %-24s 期望 %-4s 实际 %-4s"
          % ("PASS" if ok else "FAIL", name, expect, status))
    return ok


def cleanup(username, upload_id=None):
    """删掉临时账号、它的任务、以及探测用的分片目录，保持数据干净。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if upload_id:
        import shutil
        d = os.path.join(root, "uploads", "_chunks", upload_id)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    sys.path.insert(0, root)
    try:
        import storage
        conn = storage._get_conn()
        row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if row:
            uid = row[0] if not hasattr(row, "keys") else row["id"]
            # 表名：task_records / feedback / video_chunks（video_rag.db 里就这几张）
            conn.execute("DELETE FROM feedback WHERE task_id IN "
                         "(SELECT id FROM task_records WHERE user_id = ?)", (uid,))
            conn.execute("DELETE FROM task_records WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM videos WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM users WHERE id = ?", (uid,))
        conn.commit()
        print("\n已清理临时账号 %s" % username)
    except Exception as e:
        print("\n清理失败（手工删一下 %s）：%r" % (username, e))


def main():
    global BASE, MEDIA_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--media-id", type=int, default=MEDIA_ID)
    args = ap.parse_args()
    BASE, MEDIA_ID = args.base.rstrip("/"), args.media_id

    anon = session()
    me = session()
    other = session()

    print("=" * 58)
    print("未登录")
    print("=" * 58)
    case("未登录 ask", 401, "POST", "/api/ask", anon,
         payload={"question": "测试", "mediaId": MEDIA_ID})
    case("未登录 ask_agent", 401, "POST", "/api/ask_agent", anon,
         payload={"question": "测试", "mediaId": MEDIA_ID})
    case("未登录 agent_stream", 401, "POST", "/api/ask_agent_stream", anon,
         payload={"question": "测试", "mediaId": MEDIA_ID})
    case("未登录 videos", 401, "GET", "/api/videos", anon)
    case("未登录 stats", 401, "GET", "/api/stats", anon)
    case("未登录 upload", 401, "POST", "/api/upload/init", anon,
         payload={"filename": "x.mp4"})
    case("未登录 video-status", 401, "GET", "/api/video-status?id=%d" % MEDIA_ID, anon)
    case("未登录 feedback", 401, "POST", "/api/feedback", anon,
         payload={"task_id": 1, "rating": 1})
    case("未登录 playback", 404, "GET", "/api/playback?id=%d" % MEDIA_ID, anon)

    username = "__probe_%d" % int(time.time())
    password = "probe-123456"
    print()
    print("=" * 58)
    print("临时账号 %s（登录后访问别人的资源）" % username)
    print("=" * 58)
    case("注册", 200, "POST", "/api/register", anon,
         payload={"username": username, "password": password})
    case("登录", 200, "POST", "/api/login", me,
         payload={"username": username, "password": password})
    case("登录后 me", 200, "GET", "/api/me", me)
    case("他人媒体 ask", 404, "POST", "/api/ask", me,
         payload={"question": "测试", "mediaId": MEDIA_ID})
    case("他人媒体 ask_agent", 404, "POST", "/api/ask_agent", me,
         payload={"question": "测试", "mediaId": MEDIA_ID})
    case("他人媒体 stream", 404, "POST", "/api/ask_agent_stream", me,
         payload={"question": "测试", "mediaId": MEDIA_ID})
    case("他人 playback", 404, "GET", "/api/playback?id=%d" % MEDIA_ID, me)
    case("他人 playback(name)", 404, "GET", "/api/playback?name=demo.mp4", me)
    case("他人 video-status", 404, "GET", "/api/video-status?id=%d" % MEDIA_ID, me)
    case("他人 feedback", 404, "POST", "/api/feedback", me,
         payload={"task_id": 1, "rating": 1})
    case("playback 无参", 400, "GET", "/api/playback", me)

    print()
    print("=" * 58)
    print("上传通道（uploadId 绑定会话 + 路径穿越）")
    print("=" * 58)
    case("chunk 路径穿越", 404, "POST", "/api/upload/chunk", me,
         form={"uploadId": "../../etc", "index": "0"})
    case("chunk 非法格式", 404, "POST", "/api/upload/chunk", me,
         form={"uploadId": "zzzz", "index": "0"})
    init_status, _ct, init_body = call(me, "POST", "/api/upload/init",
                                       payload={"filename": "probe.mp4"})
    upload_id = None
    if init_status == 200:
        try:
            upload_id = json.loads(init_body).get("uploadId")
        except Exception:
            pass
    if upload_id:
        # 第二个账号：拿到别人的 uploadId 也不能续写
        other_name = username + "_b"
        case("注册第二个账号", 200, "POST", "/api/register", anon,
             payload={"username": other_name, "password": password})
        case("登录第二个账号", 200, "POST", "/api/login", other,
             payload={"username": other_name, "password": password})
        case("他人 uploadId status", 404, "GET",
             "/api/upload/status?uploadId=%s" % upload_id, other)
        case("他人 uploadId complete", 404, "POST", "/api/upload/complete",
             other, payload={"uploadId": upload_id})
    else:
        ROWS.append((False, "拿 uploadId", 200, init_status))
        print("FAIL 拿 uploadId 失败，跳过跨用户分片用例")

    print()
    print("=" * 58)
    print("本人资源")
    print("=" * 58)
    case("本人 videos", 200, "GET", "/api/videos", me)
    case("本人 stats", 200, "GET", "/api/stats", me)

    cleanup(username, upload_id)
    cleanup(username + "_b")

    bad = [r for r in ROWS if not r[0]]
    print()
    print("-" * 58)
    print("%d/%d 通过" % (len(ROWS) - len(bad), len(ROWS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
