FROM python:3.11-slim

# HF_ENDPOINT：容器里首次下载 whisper 模型走国内镜像，直连 huggingface 基本拉不动
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# 先拷依赖清单再装包：只改代码时这一层走缓存，不用重装
COPY requirements.txt requirements-asr.txt ./
# 默认走清华源（容器内直连 pypi.org 经常超时）；
# 海外机器或内网镜像可 build 时用 --build-arg PIP_INDEX_URL=... 覆盖
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# WITH_ASR=1 装 faster-whisper（镜像大 400MB 左右，但上传的视频能自动解析）
# 只做问答演示可以 build 时传 --build-arg WITH_ASR=0
ARG WITH_ASR=1
RUN if [ "$WITH_ASR" = "1" ]; then \
        pip install -i "$PIP_INDEX_URL" -r requirements-asr.txt; \
    else \
        pip install -i "$PIP_INDEX_URL" -r requirements.txt; \
    fi

COPY . .

EXPOSE 8001

# 不装系统 ffmpeg：requirements 里的 imageio-ffmpeg 自带静态二进制，
# pipeline._ffmpeg() 本来就优先用它。省掉 apt-get 一层，
# 也避免 deb.debian.org 在国内把构建拖成十几分钟。
# 健康检查用 Python 发请求，省掉 curl。
HEALTHCHECK --interval=15s --timeout=8s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=5).status == 200 else 1)"

CMD ["python", "app.py"]
