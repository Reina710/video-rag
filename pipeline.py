# -*- coding: utf-8 -*-
"""
自动解析流水线：上传完成后，后台跑 ASR → OCR → 向量化 → 入库。

为什么需要：
    原来这三步是三个 CLI 脚本（video_to_text_local.py / ocr_frames.py / qdrant_store.py），
    路径写死、只能手动跑；上传接口只把文件丢进 MinIO，还要人工执行命令。
    本模块把三步抽成函数，由 app.py 在上传完成后入队执行，
    并把「转写中 / 识别中 / 建索引中 / 完成 / 失败」写回 videos.status。

设计取舍：
    - 单后台线程 + 队列，不引入消息队列：单机部署够用，也讲得清"为什么异步"。
    - faster-whisper 模型懒加载并缓存（484MB，不能每个任务重载）。
    - 每个视频的中间产物落 uploads/<video_id>/，失败时能直接排查。
    - 失败不抛到 Web 层：写 status=failed + progress，接口照常返回。
    - 重试交给 OpenAI SDK 的 max_retries（指数退避）；Qdrant 单独包一层重试。
"""

import base64
import json
import os
import queue
import subprocess
import threading
import time
import uuid

import storage
import settings
import asr_lexicon


# Windows 上 httpx 会读系统代理设置（注册表里的 Internet Options），
# 连 127.0.0.1 的 Qdrant/MinIO 也会被代理拦成 502。把本地地址加进 NO_PROXY。
_no_proxy = os.environ.get("NO_PROXY", "")
if "127.0.0.1" not in _no_proxy:
    os.environ["NO_PROXY"] = ",".join(filter(None, [_no_proxy, "127.0.0.1", "localhost"]))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
COLLECTION = "video_sentences"
VECTOR_SIZE = 1024
OCR_STEP_SEC = 30


# ------------------------------------------------------------
# 懒加载：模型 / 客户端
# ------------------------------------------------------------
_WHISPER = None
_OPENAI = None


def _model_path():
    """模型位置：环境变量 > 项目内 models/ 目录 > 让库按名字自己下。

    容器里由 compose 注入 WHISPER_MODEL_PATH=/models/small（挂载卷，重建镜像不重下）。
    本地放 ./models/ 也能直接用，不再依赖某台机器上的绝对路径。
    """
    env = os.environ.get("WHISPER_MODEL_PATH", "")
    if env and os.path.exists(env):
        return env
    local = os.path.join(BASE, "models")
    if os.path.exists(local):
        return local
    return "small"


def _whisper():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        _WHISPER = WhisperModel(_model_path(), device="cpu", compute_type="int8")
    return _WHISPER


def _client():
    """DashScope 兼容客户端。max_retries 由 SDK 做指数退避重试。"""
    global _OPENAI
    if _OPENAI is None:
        from openai import OpenAI
        _OPENAI = OpenAI(
            api_key=os.environ.get("DASHSCOPE_KEY", "") or "placeholder",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60,
            max_retries=3,      # 失败重试：超时/5xx 指数退避，最多 3 次
        )
    return _OPENAI


def _ffmpeg():
    """优先用 imageio-ffmpeg 自带的，其次用系统 PATH 里的 ffmpeg。"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    return "ffmpeg"


def _retry(fn, attempts=3, base_sleep=1.0):
    """给没有内置重试的调用（Qdrant）包一层指数退避。"""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < attempts - 1:
                time.sleep(base_sleep * (2 ** i))
    raise last


# ------------------------------------------------------------
# 第一步：ASR
# ------------------------------------------------------------
def _extract_wav(video_path, out_wav):
    """用 ffmpeg 抽出 16kHz 单声道 wav，绕开 faster-whisper 对 av 的解码依赖。

    为什么要自己抽音频：faster-whisper 默认用 PyAV 解码，而 av 19 改了
    `open()` 的参数名，会报 `metadata_errors` 不兼容。先转成 numpy 数组再喂给
    transcribe()，就走不到那条解码路径，任何 av 版本都能跑。
    """
    subprocess.run(
        [_ffmpeg(), "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_wav],
        capture_output=True,
    )
    if not os.path.exists(out_wav):
        raise RuntimeError("ffmpeg 抽音频失败")


def _load_wav(path):
    """读 16-bit PCM wav，返回 [-1,1] 的 float32 单声道数组。"""
    import wave

    import numpy as np

    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("只支持 16-bit PCM wav")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def transcribe(video_path):
    """本地 faster-whisper 转写，返回 [{start, end, text}]（毫秒）。"""
    model = _whisper()
    import tempfile
    wav = os.path.join(tempfile.gettempdir(), "vr_%s.wav" % uuid.uuid4().hex[:8])
    try:
        _extract_wav(video_path, wav)
        audio = _load_wav(wav)
        hot = asr_lexicon.hotwords()
        kwargs = dict(language="zh", vad_filter=False, beam_size=5,
                      condition_on_previous_text=False)
        if hot:
            # 热词偏置：把专有名词（番剧名、人名）提前告诉解码器。
            # 只影响新转写，已经索引过的视频不会重跑。
            kwargs["hotwords"] = hot
        try:
            segments, _info = model.transcribe(audio, **kwargs)
        except TypeError:
            # 老版本 faster-whisper（<1.0）没有 hotwords 参数，退回不加热词
            kwargs.pop("hotwords", None)
            segments, _info = model.transcribe(audio, **kwargs)
        out = []
        for seg in segments:
            # 输出层纠错：小模型对热词不一定买账，这里按错字表再兜一层
            text = asr_lexicon.correct((seg.text or "").strip())
            if text:
                out.append({
                    "start": int(seg.start * 1000),
                    "end": int(seg.end * 1000),
                    "text": text,
                })
        return out
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass


# ------------------------------------------------------------
# 第二步：抽帧 + dHash 去重 + OCR
# ------------------------------------------------------------
def _dhash(image, hash_size=8):
    img = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(img.getdata())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            base = row * (hash_size + 1) + col
            bits.append(1 if pixels[base] > pixels[base + 1] else 0)
    return bits


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _video_duration(ffmpeg, video_path):
    probe = subprocess.run(
        [ffmpeg, "-i", video_path],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    for line in probe.stderr.splitlines():
        if "Duration:" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def _ocr_image(client, path):
    """识别一张图里的文字。SDK 已带重试，这里只兜异常。"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        model="qwen-vl-ocr",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": "请仅输出图像中的文本内容。"},
            ],
        }],
    )
    return (resp.choices[0].message.content or "").strip()


def ocr_video(video_path, video_id, step_sec=OCR_STEP_SEC):
    """每 step_sec 抽一帧 → dHash 去重 → OCR，返回 [{start, text}]。"""
    from PIL import Image

    ffmpeg = _ffmpeg()
    frames_dir = os.path.join(UPLOAD_DIR, str(video_id), "frames")
    os.makedirs(frames_dir, exist_ok=True)

    duration = _video_duration(ffmpeg, video_path)
    kept = []
    last_hash = None
    t = 0.0
    idx = 0
    while t < duration:
        out = os.path.join(frames_dir, "f_%04d.jpg" % idx)
        subprocess.run(
            [ffmpeg, "-ss", str(t), "-i", video_path, "-frames:v", "1",
             "-q:v", "2", "-update", "1", "-y", out],
            capture_output=True,
        )
        if os.path.exists(out) and os.path.getsize(out) > 0:
            h = _dhash(Image.open(out))
            if last_hash is None or _hamming(h, last_hash) > 5:
                kept.append((t, out))
                last_hash = h
            else:
                os.remove(out)      # 和上一帧几乎一样，删掉省 OCR 钱
        t += step_sec
        idx += 1

    client = _client()
    results = []
    for (t, path) in kept:
        try:
            text = _ocr_image(client, path)
        except Exception:
            continue        # 单帧失败跳过，不中断整条流水线
        if text and len(text) >= 2:
            results.append({"start": int(t * 1000), "text": text})
    return results


# ------------------------------------------------------------
# 第三步：向量化 + 入库
# ------------------------------------------------------------
def _embed_one(client, text):
    resp = client.embeddings.create(model="text-embedding-v3", input=text)
    return resp.data[0].embedding


def _index(video_id, asr, ocr):
    """把 ASR + OCR 双分支写进 Qdrant，payload 带 media_id 便于按视频过滤。"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(**settings.qdrant_kwargs())
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    items = [(s["start"], s["end"], s["text"], "asr") for s in asr]
    items += [(o["start"], o["start"] + OCR_STEP_SEC * 1000, o["text"], "ocr") for o in ocr]

    oai = _client()
    points = []
    for i, (start, end, text, source) in enumerate(items):
        points.append(PointStruct(
            id=int(video_id) * 1_000_000 + i,
            vector=_embed_one(oai, text),
            payload={
                "start": start,
                "end": end,
                "text": text,
                "source": source,
                "media_id": int(video_id),
            },
        ))

    for i in range(0, len(points), 100):
        batch = points[i:i + 100]
        _retry(lambda: client.upsert(collection_name=COLLECTION, points=batch))
    return len(points)


# ------------------------------------------------------------
# 编排 + 队列
# ------------------------------------------------------------
def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process(video_id, video_path):
    """一条视频的完整解析。状态一路写回 videos 表，失败不抛给 Web 层。"""
    work = os.path.join(UPLOAD_DIR, str(video_id))
    try:
        storage.update_video(video_id, status="processing", progress="转写中")
        asr = transcribe(video_path)
        _write_json(os.path.join(work, "sentences.json"), asr)

        storage.update_video(video_id, progress="识别画面文字")
        ocr = ocr_video(video_path, video_id)
        _write_json(os.path.join(work, "frames_ocr.json"), ocr)

        storage.update_video(video_id, progress="建向量索引")
        n = _index(video_id, asr, ocr)

        # 建完索引顺手生成分段摘要（长视频"全片目录"），之后问答就不用现算
        storage.update_video(video_id, progress="生成分段摘要")
        try:
            import retrieval
            retrieval.refresh_corpus()      # 刚入的向量要刷新语料缓存
            import chunks as chunks_mod
            chunks_mod.build(video_id, media_id=video_id, force=True)
        except Exception as e:
            # 摘要失败不影响主流程：问答时会按需再生成
            print("[pipeline] 分段摘要失败:", type(e).__name__)

        storage.update_video(video_id, status="done",
                             progress="完成（%d 条证据）" % n)
        print("[pipeline] video %s done: %d asr + %d ocr = %d points"
              % (video_id, len(asr), len(ocr), n))
    except Exception as e:
        storage.update_video(video_id, status="failed",
                             progress="失败：%s" % type(e).__name__)
        print("[pipeline] video %s failed: %s: %s"
              % (video_id, type(e).__name__, e))


_JOBQ = queue.Queue()
_worker_started = False


def _worker():
    while True:
        item = _JOBQ.get()
        if item is None:
            break
        video_id, video_path = item
        process(video_id, video_path)


def enqueue(video_id, video_path):
    """把一条视频的解析丢进后台队列，立即返回。"""
    global _worker_started
    if not _worker_started:
        _worker_started = True
        threading.Thread(target=_worker, daemon=True).start()
    _JOBQ.put((video_id, video_path))
