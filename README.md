# AI视频分析平台

面向长视频内容理解的可核验 RAG 问答系统。先把语音（ASR）与画面文字（OCR）抽成带时间戳的文本证据，再做文本检索与证据约束生成，回答里的每条结论都能回到原文和对应的那一秒。

**范围**：多模态只在**抽取**（音频、画面两路）和**证据溯源**（每条证据标注来源与时间戳）两层。音画入库前已转成文字，因此**检索层是单模态文本检索**——BM25 / 向量 / RRF 这一套与纯文本 RAG 相同，不支持「用截图直接找片段」这类跨模态检索；动作、表情、镜头运动、BGM 等非文字信息不在覆盖范围内。

## 核心能力

### 多模态抽取与时序对齐

- **双分支抽取**：faster-whisper 本地转写语音（ASR），qwen-vl-ocr 识别画面文字（OCR），两条分支按时间戳对齐。
- **来源可溯**：每条证据带 `source` 字段区分「语音」与「画面字幕」，回答时明确标注证据来自哪个时间点、哪条通道。

### 带证据约束的 AgentLoop

回答不依赖模型自由发挥，而是走 Planner → Executor → Critic 的闭环：

- **Planner** 将问题拆解为可执行目标；
- **Executor** 生成结构化的结论、证据与建议，每条结论绑定带时间戳的证据；
- **Critic** 检查目标覆盖与证据完整性，不通过时定向补充检索，最多两轮。

### 证据程序化核验（代码兜底）

除 Critic 的语义判断外，另有独立的程序化核验：逐条反查证据的时间戳覆盖率与原文支持率，拦截「时间戳越界」「原文不存在」这类软性错误。核验逻辑与生成逻辑解耦，避免模型自说自话。

### 混合检索（三段式）+ 上下文扩窗

向量 + BM25 双路召回 → RRF 合并 → **上下文扩窗** → rerank 精排。ASR / OCR 各自跑一遍，
避免 30+ 条画面文字被 400+ 条语音淹没。

**上下文扩窗**解决的是视频特有的问题：ASR 一句只有 2~3 秒，
「本季最强的」和它的答案「碧蓝之海」被切在两行。命中后把同源相邻句
（前 8 秒 / 后 20 秒）合并成一段再交给模型，结论和宾语才在一起。

两个顺序上的坑（都是实测踩出来的）：

- **必须先扩窗再 rerank**：反过来的话，短句「本季最强的」会因为"它没回答问题"被精排掉，
  而它恰恰是答案的前半句。
- **扩窗后要按原排名还原顺序**：否则合并会按时间排序，`limit` 取到的是"最早"而不是"最相关"。

**时间戳锚点（`anchorMs`）**：扩窗后 `start/end` 只表示窗口范围，证据时间戳和点击跳转统一用
`anchorMs`（命中那一句的起点）。最早我直接用窗口起点，结果 OCR 帧首尾相接（每帧 30 秒）、
窗口**链式合并**成 30~170 秒的一大坨：显示 `00:30`，内容其实在 `01:30`，点过去完全对不上。
现在有三条硬约束：

- 只收「起点落在窗口内」的片段，不是"有重叠就收"（否则边界帧会被拉进来）
- 窗口总长封顶 40 秒
- OCR 一帧本身就覆盖 30 秒，只取命中帧自身，不再前后扩

另外证据时间戳还有一层**确定性兜底**：按引文原文匹配到所在片段，再用该片段的 `anchorMs`，
不依赖模型是否听话。

`--compare` 实测（18 题，句子级）：纯向量 hit@1 88.9% → 加融合 **94.4%**。
初版这里曾出现"融合反而掉到 75%"，根因是 BM25 这一路的两个前提没校准（繁简、长度归一），
不是融合本身的问题，详见下面「离线评测」。
`--content` 实测：答案真的落进证据的比例，句子级 43.1% → 扩窗后 **94.1%**。

### 长视频分段摘要（解决「只看开头」）

视频按 5 分钟切块，每块用 LLM 生成摘要 + 关键词，存 SQLite（`video_chunks` 表），
问答时作为「全片目录」同时喂给 Planner / Executor / Critic。

**为什么需要**：向量检索最容易召回的是开场总述，证据会**全挤在前几分钟**——长视频 RAG 的
「时间坍缩」。我先试过"时间覆盖约束"，治标：模型看到的还是零散句子，它不知道自己漏了什么。
分段目录让模型先知道全片讲了什么，再决定深入哪一段，缺证据时也能诚实说"没检索到"。

实测（16 分钟视频）：加分段摘要前，答案只覆盖 0–7 分钟；加之后覆盖 0–16 分钟，
补出了 08:26《二十世纪电器目录》和 15:14《再见琵琵》这两段原本完全没进上下文的内容。

### 多用户与存储

- **多用户**：注册 / 登录 / 登出，登录会话存 Redis，任务记录按用户隔离。
- **接口鉴权**：数据接口分两层——先看有没有登录，再校验这条记录是不是你的
  （视频 / 上传会话 / 任务记录）。换一个 id 读不到别人的内容，越权统一回 404 而不是 403，
  避免被用来枚举资源是否存在；`uploadId` 绑定创建者并做格式校验，挡住 `../` 穿越。
- **存储抽象层**：SQLite（本地开发）与 MySQL（生产）双后端，SQL 层一致，改环境变量即可切换。
- **异步落库**：写库走后台队列，不阻塞问答响应。
- **反馈闭环**：用户对回答点满意 / 不满意，累计计算用户接受率。

## 工程能力

| 能力 | 实现 |
|---|---|
| 自动解析 | 上传完成后后台线程自动跑 ASR → OCR → 向量化，进度写回 `videos.status/progress`，前端轮询显示 |
| 分片上传 | 5MB 分片，`init / chunk / status / complete` 四个接口；断点续传先查已传分片，只补缺的 |
| 失败重试 | OpenAI 客户端 `max_retries=3`（指数退避）；Qdrant upsert 单独包一层重试 |
| 健康检查 | `GET /health` 返回 redis / minio / qdrant 依赖状态 |
| 上传去重 | 文件 SHA256，同一视频重复上传复用已有索引，不重复烧 ASR/OCR |
| 多视频隔离 | 检索按 `media_id` 过滤，问 A 视频不会捞到 B 视频的证据（时间戳跳转才对得上） |
| 多轮追问改写 | 追问（「那第二部呢」）先补全成完整问题再检索；无历史或问题已自足时跳过，不花 LLM |

## 离线评测

`evaluate.py` 内置黄金评测集（正向题 + 对抗题），对每条回答计算结构化指标：

- `structuredValid` 结构完整性
- `timestampCoverageRate` 时间戳覆盖率
- `evidenceSupportRate` 证据支持率
- `claimEvidenceSupportRate` 结论—证据绑定率
- `criticPassed` Critic 是否通过
- `keywordCoverage` 关键信息命中率

对抗题专门考察「问视频里没有的内容时能否拒绝回答」，用于验证防幻觉能力。

**检索层评测**（`eval_retrieval.py`）：端到端指标只说明最终答案好不好，定位不到问题出在
「检索没捞到」还是「生成没写好」。这一层用人工标定的**黄金时间区间**把检索单独拎出来量：

| 指标 | 含义 |
|---|---|
| `hit@K` | Top-K 里有没有落在正确时间区间的证据 |
| `recall@K` | 一个问题有多个正确区间时，被覆盖的比例 |
| `MRR` | 第一条命中证据排在第几名的倒数 |

**三种配置对比**（18 题，同一套题、同一批数据）：

| 配置 | hit@1 | hit@3 | hit@5 | MRR@5 | 中位耗时 |
|---|---|---|---|---|---|
| ① 纯向量 | 88.9% | 94.4% | 100% | 0.928 | 218ms |
| ② 向量 + BM25 + RRF | **94.4%** | 94.4% | 100% | 0.958 | 206ms |
| ③ 再加 rerank | 94.4% | 94.4% | 100% | 0.958 | 384ms |

初版这组数字是「纯向量 87.5% → 融合 75% → rerank 100%」，看起来融合有害、只有 rerank 有用。
拆开看掉点的根因不在融合，在 BM25 这一路的两个前提被直接照搬了：

1. **繁简没统一**：ASR 输出繁体、用户提问是简体，「漫畫」和「漫画」在 BM25 里是两个词，
   字面召回直接归零（8 题里只有 7/8 能召回）。embedding 对繁简不敏感，于是变成
   「向量能召回、BM25 全是噪声」，RRF 把错误项抬进 Top-1。
2. **长度归一的假设不成立**：b=0.75 的假设是「文档越长，命中的词越可能是巧合」。
   但 ASR 一行只有 3~4 个词，长度差异来自语速而不是信息量。b=0.75 会把
   「我最愛的漫畫」这种 7 字短句的权重放大 5 倍，压过真正有上下文的片段。改为 b=0.4。

两处修完后，同一 8 题上 hit@1 从 75.0% 回到 **100%**。融合不是"锦上添花"，
它的作用是补字面召回的短板；这一路可信时融合就该涨。

**rerank 值不值**：18 题上 rerank 对 hit@1 / MRR 都是 0 增量，代价 +178ms。
但换个口径结论就变了，见下面的内容级评测。

> 评测测的是**句子级检索**（不扩窗）：28 秒的窗口几乎什么都能和黄金区间重叠，
> 指标会全部饱和，扩窗的收益在这个口径下测不出来。

**内容级评测**（`--content`）：hit@K 只回答"名次对不对"，不回答"拿到的那块里有没有答案"。
一句「本季最强的」可以排第一名，名次满分，但宾语在下一句 —— 块里没有答案。

| 证据形态 | 答案覆盖率 | 完全命中 | 中位耗时 |
|---|---|---|---|
| 句子级（不扩窗） | 43.1% | 41.2% | 403ms |
| 扩窗（前 8s / 后 20s） | **94.1%** | **94.1%** | 376ms |

扩窗把"答案真的在证据里"的比例从 43% 拉到 94%，代价接近零（本地拼接相邻句）。
rerank 在这个口径下才显出价值：窗口 + RRF 88.2% → 加 rerank **94.1%**，代价约 170ms。

同一个系统，两套口径给出两个不同的"要不要 rerank"的答案：名次口径 0 增量，
内容口径 +5.9 个点。所以报指标必须同时说清 K 和证据形态，**跨口径的数字不能直接相减**。

## 技术栈

| 层次 | 技术 | 用途 |
| :--- | :--- | :--- |
| Web | Vue 3、Vite、Flask | 前端问答工作台与后端 API |
| 多模态抽取 | faster-whisper、qwen-vl-ocr | 语音转写与画面文字识别（抽取后统一为文本） |
| 向量检索 | Qdrant、text-embedding-v3 | 语义检索证据片段 |
| 生成与推理 | DeepSeek | 回答生成与 AgentLoop 三角色 |
| 存储 | SQLite / MySQL、MinIO、Redis | 业务数据、视频对象、会话 |
| 异步 | 线程队列 | 落库削峰 |

Redis 除会话外，还承担两个真实用途：热点问题缓存（同一问题重复问直接返回，省 embedding 与 LLM 的 API 成本）与滑动窗口限流（防止刷接口烧 API 钱）。

## 本地运行

### 环境要求

- Python 3.10+
- Docker（用于 MySQL / Redis / Qdrant / MinIO）
- ffmpeg（faster-whisper 解码依赖）

```bash
pip install -r requirements.txt        # 服务 + 检索
pip install -r requirements-asr.txt    # 需要上传视频/转写时再加（含 faster-whisper）
```

### 1. 准备配置

```bash
cp .env.example .env
```

编辑 `.env`，填入 `DASHSCOPE_KEY` 与 `DEEPSEEK_KEY`，并按需修改 MySQL、MinIO 等密码。`.env` 已在 `.gitignore` 中，不会提交到仓库。

配置读取统一走 `settings.py`：**环境变量优先，其次项目根目录的 `.env`**。部署时直接注入环境变量即可，不需要改代码。

### 2. 启动中间件

一条命令起全套（app + MySQL + Redis + Qdrant + MinIO）：

> 镜像里不带前端产物（`client/dist` 不进仓库）。第一次启动前先在 `client/` 跑一次
> `npm ci && npm run build`（见第 3 节），否则容器起来了但页面是空的。

```bash
docker compose up -d     # 首次会构建镜像
docker compose ps        # 等 video-rag-app 变成 healthy
```

不需要「上传视频自动解析」、只想跑问答时，在 `.env` 里设 `WITH_ASR=0` 再构建，镜像能小一半。

只想起中间件、app 在宿主机跑：

```bash
docker compose up -d mysql redis qdrant minio
```

> 容器里 `app` 的中间件地址由 compose 注入成服务名（`qdrant` / `redis` / `minio`）。
> 所以这类地址在代码里不能硬编码——容器内的 `127.0.0.1` 指的是容器自己。

两个部署时的细节：

- **whisper 模型放在命名卷里**（`video_rag_whisper` → 容器 `/models/small`）。卷为空时 faster-whisper 会自动下载；
  本机已有缓存可以直接灌进去，重建镜像不会重下：`docker cp <模型目录> video-rag-app:/models/small`。
- 机器上若已存在同名数据卷（早期用 `docker run` 起的 qdrant / minio），compose 会警告
  `was not created by Docker Compose` 并继续复用。**数据不会被清空**，可以忽略；全新机器上没有这条警告。

### 3. 启动服务

```bash
python app.py
```

访问 `http://127.0.0.1:8001`。服务会直接返回 `client/dist` 的 Vue 构建产物；开发前端时在 `client` 目录运行 `npm run dev`，Vite 会把 `/api` 请求代理到 8001。

首次跑或改过前端后需要构建一次（`client/dist` 不进仓库）：

```bash
cd client
npm ci
npm run build
```

### 4. 准备视频

上传视频后会自动解析（ASR → OCR → 向量化），进度在左侧视频列表显示。

**ASR 词表**：转写质量靠两层，都在 `data/` 下，改完重启生效。

| 文件 | 作用 | 覆盖方式 |
|---|---|---|
| `hotwords.txt` | 热词偏置，一行一个词，转写时提前告诉解码器 | `ASR_HOTWORDS="词1 词2"` |
| `asr_fixups.txt` | 输出层纠错，`错=对` 一行一条 | `ASR_FIXUPS=0` 关闭 |

实测同一段 2 分钟音频、17 个目标专有名词：不带热词命中 2 个，带热词命中 4 个，
但两版文本差异 11.2%——热词是偏置不是保证，会让部分词变好、部分变差。
纠错表要谨慎：写错一条规则等于主动制造错误，加规则前先用 `ASR_FIXUPS=0` 看原文。

> 热词只影响新上传的转写；同内容视频命中去重，不会重跑 ASR。

下面的 CLI 仍保留，用于批量/离线处理（参数是视频路径）：

```bash
python video_to_text_local.py <视频路径>   # 转写语音 -> sentences.json
python ocr_frames.py <视频路径>            # 识别画面文字 -> frames_ocr.json
python qdrant_store.py                     # 写入 Qdrant
```

### 5. 运行离线评测

```bash
python evaluate.py                      # 端到端：结构化 + 证据核验 + 防幻觉
python eval_retrieval.py --compare       # 检索层：三种配置 × hit@1/3/5 + MRR + 耗时
python eval_retrieval.py --content       # 内容级：扩窗前后答案有没有真的进证据
python -m unittest discover tests -v    # 后端单元测试（检索排序/合并逻辑）
python scripts/verify_auth.py           # 接口鉴权：未登录 / 越权 / 本人 28 个用例
```

端到端结果写入 `evaluation_report.json`。检索评测直接打印汇总表。

## 目录结构

```text
video-rag
├── app.py                    # Flask 入口：20 个路由（问答 / 上传 / 认证 / 反馈 / 健康检查）
├── settings.py               # 配置读取：环境变量 > 项目内 .env
├── asr_lexicon.py            # ASR 词表：热词偏置 + 输出层错字纠正
├── agent_loop.py             # Planner / Executor / Critic 主循环
├── retrieval.py              # 检索层：向量 + BM25 → RRF → 扩窗 → rerank
├── chunks.py                 # 长视频分段摘要（5 分钟一段的"全片目录"）
├── query_rewrite.py          # 多轮追问改写（把「那第二部呢」补全）
├── evidence_verify.py        # 证据程序化核验
├── pipeline.py               # 上传后的后台流水线：ASR → OCR → 向量化 → 状态回写
├── storage.py                # SQLite / MySQL 存储抽象层
├── evaluate.py               # 端到端评测
├── eval_retrieval.py         # 检索层评测（hit@K / recall@K / MRR / 内容级覆盖率）
├── tests/                    # 后端单测 22 个（RRF / 扩窗 / 分词 / 繁简 / 命中判定）
├── scripts/verify_auth.py    # 接口鉴权回归：28 个用例，跑完自动清理
├── client/                   # Vue 3 + Vite 前端（含 28 个 vitest 用例）
├── requirements.txt          # 服务 + 检索依赖
├── requirements-asr.txt      # 额外安装 faster-whisper（转写用）
├── Dockerfile                # app 镜像（基础层不含 ffmpeg/curl，静态 ffmpeg 由 pip 包提供）
├── docker-compose.yml        # 一套起 app + MySQL + Redis + Qdrant + MinIO
├── data/                     # ASR 词表：hotwords.txt / asr_fixups.txt
├── .env.example              # 配置模板
└── .github/workflows/ci.yml  # CI：语法检查 + 后端单测 + 前端测试
```

离线脚本（`pipeline.py` 的前身，只用于批量处理，服务运行不依赖）：

```text
video_to_text_local.py        # 本地 faster-whisper 转写
ocr_frames.py                 # 抽帧 + OCR
qdrant_store.py               # 手工重建向量库（会先清空 collection）
```

## License

MIT
