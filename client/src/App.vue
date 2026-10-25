<script setup>
import { ref, onMounted } from 'vue'
import { useAuth } from './useAuth'
import { useChat } from './useChat'
import { useVideos } from './useVideos'
import { fmtMs } from './format'

// ---------- 组合式函数：状态与逻辑都抽到各模块了 ----------
const auth = useAuth()
const chat = useChat()
const videoStore = useVideos()

const {
  loggedIn, username, authTab, authUsername, authPassword,
  authPassword2, authError, authSuccess, switchTab,
} = auth
const { mode, question, messages, loading, stage, stageText, stageDetail, stages, elapsed, ask } = chat
const { videos } = videoStore

// ---------- 跨模块编排 ----------
async function refreshAuth() {
  const ok = await auth.refreshAuth()
  if (ok) videoStore.loadVideos()
}

async function submitAuth() {
  const ok = await auth.submitAuth()
  if (ok && auth.loggedIn.value) videoStore.loadVideos()
}

async function logout() {
  await auth.logout()
  chat.clear()
  videoStore.clear()
  currentVideo.value = null
}

// ---------- 视频 ----------
const playerRef = ref(null)
const playerBoxRef = ref(null)
const videoInputRef = ref(null)
const currentVideo = ref(null)
const uploadText = ref('')

function pickVideo() {
  videoInputRef.value.click()
}

async function onVideoChange(e) {
  const f = e.target.files[0]
  if (!f) return
  uploadText.value = '上传中 0%'
  const d = await videoStore.upload(f, (done, total) => {
    uploadText.value = '上传中 ' + Math.round((done / total) * 100) + '%'
  })
  if (d.error) {
    alert('上传失败：' + d.error)
    uploadText.value = ''
    return
  }

  await videoStore.loadVideos()
  const picked = videos.value.find((v) => v.id === d.videoId) || {
    filename: d.filename,
    object_name: d.objectName,
    status: d.deduplicated ? 'done' : 'processing',
  }
  selectVideo(picked)

  if (d.videoId) {
    uploadText.value = d.deduplicated ? '视频已存在，已就绪' : '解析中…'
    videoStore.pollStatus(d.videoId, (v) => {
      uploadText.value = v.progress || v.status
      if (v.status === 'done') setTimeout(() => { uploadText.value = '' }, 2500)
    })
  }
  e.target.value = ''
}

function selectVideo(v) {
  if (!v) return
  currentVideo.value = v
  const player = playerRef.value
  if (player && v.object_name) {
    player.src = '/api/playback?name=' + encodeURIComponent(v.object_name)
    player.load()
  }
}

function onPlayerError() {
  flashHint('这个视频的文件不在存储里（可能已被清理），请重新上传')
}

function jumpTo(ms) {
  const seconds = Number(ms) / 1000
  if (!Number.isFinite(seconds)) return
  const player = playerRef.value
  if (!player || !player.src) {
    flashHint('请先在左侧选择或上传视频')
    return
  }
  if (player.readyState === 0) {
    player.addEventListener('loadedmetadata', () => jumpTo(ms), { once: true })
    return
  }
  const duration = player.duration
  const maxTime = Number.isFinite(duration) ? Math.max(0, duration - 0.1) : seconds
  player.currentTime = Math.min(Math.max(0, seconds), maxTime)
  player.play().catch(() => {})
  player.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  flashPlayer()
}

function flashPlayer() {
  const box = playerBoxRef.value
  if (!box) return
  box.classList.add('seek-flash')
  setTimeout(() => box.classList.remove('seek-flash'), 900)
}

const hint = ref('')
let hintTimer = null
function flashHint(text) {
  hint.value = text
  clearTimeout(hintTimer)
  hintTimer = setTimeout(() => { hint.value = '' }, 2200)
}

// 提问时带上当前视频 id，后端只在这个视频的证据里检索（多视频隔离）
function askWithVideo() {
  ask(currentVideo.value ? currentVideo.value.id : null)
}

// 同一时间戳 + 同一来源的多条证据合并成一行展示。
// 数据层不合并（每条证据都绑着自己的 claim，合并会破坏核验），只在展示层去重。
function groupEvidence(list) {
  const groups = []
  const index = {}
  for (const h of list || []) {
    const ts = h.timestampMs !== undefined ? h.timestampMs : (h.start || 0)
    const src = (h.source || 'asr').toLowerCase()
    const text = h.text !== undefined ? h.text : (h.content || '')
    const key = ts + '|' + src
    if (index[key] === undefined) {
      index[key] = groups.length
      groups.push({ ts, src, texts: [text] })
    } else if (text && !groups[index[key]].texts.includes(text)) {
      groups[index[key]].texts.push(text)
    }
  }
  return groups
}

// 全片目录折叠状态（按消息下标）
const openChunks = ref({})
function toggleChunks(i) {
  openChunks.value = { ...openChunks.value, [i]: !openChunks.value[i] }
}

const fmt = fmtMs
onMounted(refreshAuth)
</script>

<template>
  <!-- ================= 登录 ================= -->
  <div v-if="!loggedIn" class="auth-page">
    <div class="bg-layer"><span class="grid"></span><span class="orb o1"></span><span class="orb o2"></span></div>
    <div class="auth-card">
      <div class="auth-brand">
        <div class="logo">V</div>
        <h1>VideoRAG</h1>
        <p>长视频内容理解 · 多模态 RAG 问答</p>
      </div>

      <div class="auth-tabs">
        <button :class="{ active: authTab === 'login' }" @click="switchTab('login')">登录</button>
        <button :class="{ active: authTab === 'register' }" @click="switchTab('register')">注册</button>
      </div>

      <div class="auth-form">
        <input v-model="authUsername" type="text" placeholder="用户名" @keyup.enter="submitAuth" />
        <input v-model="authPassword" type="password" placeholder="密码（至少 6 位）" @keyup.enter="submitAuth" />
        <input v-if="authTab === 'register'" v-model="authPassword2" type="password" placeholder="确认密码" @keyup.enter="submitAuth" />
        <button class="auth-submit" @click="submitAuth">
          {{ authTab === 'login' ? '登 录' : '注 册' }}
        </button>
        <div v-if="authError" class="auth-error">{{ authError }}</div>
        <div v-if="authSuccess" class="auth-success">{{ authSuccess }}</div>
      </div>
    </div>
  </div>

  <!-- ================= 工作台 ================= -->
  <div v-else class="workspace">
    <div class="bg-layer"><span class="grid"></span><span class="orb o1"></span><span class="orb o2"></span></div>

    <header class="topbar">
      <div class="brand-mini">
        <div class="logo small">V</div>
        <div class="brand-text">
          <span class="brand-name">VideoRAG</span>
          <span class="brand-sub">多模态视频理解</span>
        </div>
      </div>
      <div class="topbar-right">
        <span class="status-chip" :class="{ busy: videos.some(v => v.status === 'processing') }">
          <span class="chip-dot"></span>
          {{ videos.some(v => v.status === 'processing') ? '解析中' : '就绪' }}
        </span>
        <span class="user-name">{{ username }}</span>
        <button class="ghost-btn" @click="logout">登出</button>
      </div>
    </header>

    <div class="body">
      <!-- 左：视频库 -->
      <aside class="panel sidebar">
        <div class="panel-head">
          <span class="panel-title">视频库</span>
          <span class="count">{{ videos.length }}</span>
        </div>

        <button class="upload-btn" @click="pickVideo"><span class="plus">＋</span> 上传视频</button>
        <input ref="videoInputRef" type="file" accept="video/*" style="display:none" @change="onVideoChange" />
        <div v-if="uploadText" class="upload-progress"><span class="mini-spinner"></span>{{ uploadText }}</div>

        <div class="video-list">
          <div v-if="!videos.length" class="empty">还没有视频<br />上传一个开始提问</div>
          <div
            v-for="v in videos"
            :key="v.id"
            class="video-item"
            :class="{ active: currentVideo && currentVideo.id === v.id }"
            @click="selectVideo(v)"
          >
            <span class="dot" :class="'dot-' + (v.status || 'uploaded')"></span>
            <div class="video-item-body">
              <div class="video-item-name">{{ v.filename }}</div>
              <div class="video-item-meta">
                {{ (v.size_bytes / 1024 / 1024).toFixed(1) }} MB
                <span v-if="v.status === 'processing'"> · {{ v.progress || '解析中' }}</span>
                <span v-else-if="v.status === 'done'"> · 已就绪</span>
                <span v-else-if="v.status === 'failed'"> · 失败</span>
              </div>
            </div>
          </div>
        </div>
      </aside>

      <!-- 中：视频舞台 -->
      <main class="stage">
        <div class="stage-head">
          <div class="stage-title">
            <span class="stage-name" :class="{ dim: !currentVideo }">{{ currentVideo ? currentVideo.filename : '未选择视频' }}</span>
            <span v-if="currentVideo && currentVideo.status === 'processing'" class="stage-state">{{ currentVideo.progress || '解析中' }}</span>
            <span v-else-if="currentVideo && currentVideo.status === 'done'" class="stage-state ok">已就绪</span>
          </div>
          <div class="mode-switch">
            <button :class="{ active: mode === 'rag' }" @click="mode = 'rag'">普通 RAG</button>
            <button :class="{ active: mode === 'agent' }" @click="mode = 'agent'">AgentLoop</button>
          </div>
        </div>

        <div class="player-shell">
          <div ref="playerBoxRef" class="player-box">
            <video ref="playerRef" controls @error="onPlayerError"></video>
            <div v-if="!currentVideo" class="player-empty">
              <div class="scan"></div>
              <div class="play-orb">▶</div>
              <div class="player-empty-text">选择左侧视频，或上传新视频</div>
            </div>
          </div>
        </div>

        <div v-if="hint" class="hint-bar">{{ hint }}</div>
      </main>

      <!-- 右：对话 -->
      <aside class="panel chat-panel">
        <div class="panel-head">
          <span class="panel-title">对话</span>
          <span class="count">{{ messages.length }}</span>
        </div>

        <div class="chat">
          <div v-if="!messages.length" class="chat-empty">
            <div class="chat-empty-icon">◈</div>
            选好视频后，在下方提问
          </div>

          <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
            <template v-if="m.role === 'bot'">
              <div v-if="m.summary" class="msg-summary">
                <span class="summary-label">总结</span>
                <span class="summary-text">{{ m.summary }}</span>
              </div>

              <div v-if="m.chunks && m.chunks.length" class="chunks">
                <div class="chunks-head" @click="toggleChunks(i)">
                  <span>全片目录（{{ m.chunks.length }} 段）</span>
                  <span class="chunks-toggle">{{ openChunks[i] ? '收起' : '展开' }}</span>
                </div>
                <div v-if="openChunks[i]" class="chunks-body">
                  <div v-for="(c, k) in m.chunks" :key="k" class="chunk">
                    <span class="chunk-time">{{ fmt(c.startMs) }}–{{ fmt(c.endMs) }}</span>
                    <span class="chunk-text">{{ c.summary }}</span>
                  </div>
                </div>
              </div>
              <template v-if="m.title">
                <div class="msg-title">{{ m.title }}</div>
                <div v-for="(c, j) in m.conclusions" :key="j" class="msg-concl">· {{ c }}</div>
                <div class="msg-critique">Critic {{ m.critique && m.critique.passed ? '通过' : '未通过' }} · {{ m.rounds }} 轮</div>
              </template>
              <template v-else>{{ m.text }}</template>

              <div v-if="m.evidence && m.evidence.length" class="hits">
                <div class="hits-label">命中证据</div>
                <div v-for="(h, j) in groupEvidence(m.evidence)" :key="j" class="hit">
                  <span class="ts" @click="jumpTo(h.ts)">[{{ fmt(h.ts) }}]</span>
                  <span class="src" :class="h.src">{{ h.src === 'ocr' ? '画面' : '语音' }}</span>
                  <span class="hit-text">{{ h.texts.join('；') }}</span>
                </div>
              </div>
            </template>
            <template v-else>{{ m.text }}</template>
          </div>

          <div v-if="loading" class="agent-progress">
            <div v-if="stages.length" class="stage-badges">
              <span v-for="(s, i) in stages" :key="i" class="stage-badge" :class="'st-' + s.key">{{ s.label }} · {{ s.detail }}</span>
            </div>
            <div class="stage-now">
              <span class="spinner"></span>
              <span class="stage-text">{{ stageText || '思考中' }}</span>
              <span v-if="stageDetail" class="stage-detail">{{ stageDetail }}</span>
              <span class="stage-elapsed">{{ String(Math.floor(elapsed / 60)).padStart(2, '0') }}:{{ String(elapsed % 60).padStart(2, '0') }}</span>
            </div>
          </div>
        </div>

        <div class="input-row">
          <input v-model="question" type="text" placeholder="例如：视频里提到了哪家公司？" @keyup.enter="askWithVideo" />
          <button :disabled="loading" @click="askWithVideo">发送</button>
        </div>
      </aside>
    </div>
  </div>
</template>

<style>
/* ============================================================
   科技感暗色主题
   - 背景：深空 + 网格 + 漂浮光斑
   - 面板：玻璃拟态（半透明 + 模糊 + 细边框）
   - 按钮：渐变 + 光晕 + 扫光动效
   ============================================================ */
:root {
  --bg: #05070d;
  --bg-2: #080c16;
  --panel: rgba(13, 19, 31, 0.72);
  --panel-solid: #0d131f;
  --card: rgba(20, 28, 44, 0.66);
  --line: rgba(120, 160, 200, 0.14);
  --line-strong: rgba(120, 160, 200, 0.26);
  --text: #e7eefb;
  --dim: #93a1b8;
  --faint: #5d6b82;
  --cyan: #22d3ee;
  --violet: #8b5cf6;
  --lime: #a3e635;
  --amber: #fbbf24;
  --red: #fb7185;
  --grad: linear-gradient(120deg, #22d3ee, #8b5cf6);
  --glow: 0 0 24px rgba(34, 211, 238, 0.35);
  --mono: "SF Mono", "JetBrains Mono", Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, #app { height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  overflow: hidden;
}

/* ---------- 背景层：网格 + 光斑 ---------- */
.bg-layer { position: fixed; inset: 0; z-index: -1; overflow: hidden; }
.bg-layer .grid {
  position: absolute; inset: -50%;
  background-image:
    linear-gradient(rgba(120, 160, 200, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(120, 160, 200, 0.05) 1px, transparent 1px);
  background-size: 46px 46px;
  transform: perspective(500px) rotateX(52deg) translateY(-18%);
  mask-image: radial-gradient(ellipse at 50% 30%, #000 0%, transparent 72%);
  animation: gridDrift 24s linear infinite;
}
@keyframes gridDrift { from { background-position: 0 0; } to { background-position: 0 460px; } }
.orb { position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.5; }
.o1 { width: 620px; height: 620px; top: -260px; left: 8%; background: radial-gradient(circle, rgba(34,211,238,0.34), transparent 65%); animation: float1 18s ease-in-out infinite; }
.o2 { width: 520px; height: 520px; bottom: -220px; right: 4%; background: radial-gradient(circle, rgba(139,92,246,0.32), transparent 65%); animation: float2 22s ease-in-out infinite; }
@keyframes float1 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(40px, 30px); } }
@keyframes float2 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-40px, -24px); } }

/* ---------- 登录 ---------- */
.auth-page { height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
.auth-card {
  position: relative; width: 100%; max-width: 410px; padding: 42px 38px 36px;
  background: var(--panel); backdrop-filter: blur(18px);
  border: 1px solid var(--line); border-radius: 20px;
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.05);
}
.auth-card::before {
  content: ""; position: absolute; inset: -1px; border-radius: 21px; padding: 1px;
  background: linear-gradient(140deg, rgba(34,211,238,0.55), transparent 40%, transparent 60%, rgba(139,92,246,0.55));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
}
.auth-brand { text-align: center; margin-bottom: 26px; }
.logo {
  width: 54px; height: 54px; border-radius: 16px; margin: 0 auto 14px;
  background: var(--grad); color: #04121a; font-size: 26px; font-weight: 800;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 34px rgba(34, 211, 238, 0.5);
  animation: breathe 3.2s ease-in-out infinite;
}
@keyframes breathe { 0%,100% { transform: translateY(0); box-shadow: 0 0 24px rgba(34,211,238,0.38); } 50% { transform: translateY(-3px); box-shadow: 0 0 40px rgba(139,92,246,0.55); } }
.auth-brand h1 { font-size: 23px; letter-spacing: 0.5px; }
.auth-brand p { margin-top: 7px; font-size: 12.5px; color: var(--faint); font-family: var(--mono); }
.auth-tabs { display: flex; gap: 8px; margin-bottom: 18px; background: rgba(255,255,255,0.03); padding: 4px; border-radius: 12px; }
.auth-tabs button {
  flex: 1; padding: 9px; border: 1px solid transparent; border-radius: 9px;
  background: transparent; color: var(--dim); font-size: 13px; cursor: pointer; transition: 0.2s;
}
.auth-tabs button.active { background: var(--grad); color: #04121a; font-weight: 700; box-shadow: 0 4px 18px rgba(34,211,238,0.35); }
.auth-form { display: flex; flex-direction: column; gap: 12px; }
.auth-form input {
  padding: 12px 15px; border: 1px solid var(--line); border-radius: 11px;
  background: rgba(8, 12, 22, 0.8); color: var(--text); font-size: 14px; outline: none; transition: 0.18s;
}
.auth-form input:focus { border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(34,211,238,0.14); }
.auth-submit {
  position: relative; overflow: hidden; margin-top: 6px; padding: 13px;
  border: none; border-radius: 11px; background: var(--grad); color: #04121a;
  font-size: 15px; font-weight: 800; letter-spacing: 1px; cursor: pointer; transition: 0.2s;
}
.auth-submit::after {
  content: ""; position: absolute; top: 0; left: -60%; width: 40%; height: 100%;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,0.55), transparent);
  transform: skewX(-20deg); animation: shine 3.4s ease-in-out infinite;
}
@keyframes shine { 0%, 62% { left: -60%; } 82%, 100% { left: 130%; } }
.auth-submit:hover { box-shadow: 0 0 30px rgba(34,211,238,0.5); transform: translateY(-1px); }
.auth-error { font-size: 12.5px; color: var(--red); text-align: center; }
.auth-success { font-size: 12.5px; color: var(--lime); text-align: center; }

/* ---------- 工作台骨架 ---------- */
.workspace { height: 100vh; display: flex; flex-direction: column; position: relative; }
.topbar {
  height: 58px; flex-shrink: 0; display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px; border-bottom: 1px solid var(--line);
  background: rgba(6, 10, 18, 0.66); backdrop-filter: blur(16px);
}
.brand-mini { display: flex; align-items: center; gap: 11px; }
.logo.small { width: 32px; height: 32px; border-radius: 10px; font-size: 15px; margin: 0; animation: none; }
.brand-text { display: flex; flex-direction: column; line-height: 1.25; }
.brand-name { font-size: 14.5px; font-weight: 800; letter-spacing: 0.6px; background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.brand-sub { font-size: 10.5px; color: var(--faint); font-family: var(--mono); }
.topbar-right { display: flex; align-items: center; gap: 14px; }
.status-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 11px; border-radius: 999px; font-size: 11.5px; font-family: var(--mono);
  border: 1px solid rgba(163,230,53,0.3); color: var(--lime); background: rgba(163,230,53,0.07);
}
.status-chip .chip-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--lime); box-shadow: 0 0 8px var(--lime); }
.status-chip.busy { border-color: rgba(251,191,36,0.35); color: var(--amber); background: rgba(251,191,36,0.08); }
.status-chip.busy .chip-dot { background: var(--amber); box-shadow: 0 0 8px var(--amber); animation: pulse 1.2s ease-in-out infinite; }
.user-name { font-size: 12.5px; color: var(--dim); }
.ghost-btn {
  padding: 6px 15px; border: 1px solid var(--line); border-radius: 9px;
  background: transparent; color: var(--dim); font-size: 12.5px; cursor: pointer; transition: 0.2s;
}
.ghost-btn:hover { border-color: var(--cyan); color: var(--cyan); box-shadow: 0 0 16px rgba(34,211,238,0.25); }

.body { flex: 1; display: flex; gap: 14px; padding: 14px; min-height: 0; }

/* ---------- 通用面板 ---------- */
.panel {
  background: var(--panel); backdrop-filter: blur(16px);
  border: 1px solid var(--line); border-radius: 16px;
  display: flex; flex-direction: column; min-height: 0;
  box-shadow: 0 16px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
}
.panel-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 16px 10px; }
.panel-title { font-size: 12px; font-weight: 700; letter-spacing: 1.2px; color: var(--dim); text-transform: uppercase; }
.count { font-size: 11px; font-family: var(--mono); color: var(--faint); background: rgba(255,255,255,0.05); border: 1px solid var(--line); padding: 1px 8px; border-radius: 999px; }

/* ---------- 左侧视频库 ---------- */
.sidebar { width: 268px; flex-shrink: 0; }
.upload-btn {
  position: relative; overflow: hidden; margin: 0 14px 12px; padding: 11px;
  border: 1px solid rgba(34,211,238,0.4); border-radius: 11px;
  background: linear-gradient(120deg, rgba(34,211,238,0.16), rgba(139,92,246,0.16));
  color: var(--cyan); font-size: 13px; font-weight: 700; cursor: pointer; transition: 0.2s;
}
.upload-btn::after {
  content: ""; position: absolute; top: 0; left: -60%; width: 45%; height: 100%;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,0.35), transparent);
  transform: skewX(-20deg); animation: shine 4s ease-in-out infinite;
}
.upload-btn:hover { background: var(--grad); color: #04121a; box-shadow: 0 0 26px rgba(34,211,238,0.5); transform: translateY(-1px); }
.upload-progress { display: flex; align-items: center; gap: 7px; margin: 0 16px 10px; font-size: 11.5px; color: var(--dim); font-family: var(--mono); }
.mini-spinner {
  width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0;
  border: 2px solid rgba(120,160,200,0.25); border-top-color: var(--cyan);
  animation: spin 0.8s linear infinite;
}
.video-list { flex: 1; overflow-y: auto; padding: 0 10px 12px; display: flex; flex-direction: column; gap: 4px; }
.empty { text-align: center; color: var(--faint); font-size: 12px; padding: 26px 0; line-height: 1.9; }
.video-item {
  display: flex; gap: 10px; padding: 11px 12px; border-radius: 11px; cursor: pointer;
  border: 1px solid transparent; transition: 0.18s;
}
.video-item:hover { background: rgba(255,255,255,0.04); border-color: var(--line); transform: translateX(2px); }
.video-item.active {
  background: linear-gradient(120deg, rgba(34,211,238,0.12), rgba(139,92,246,0.1));
  border-color: rgba(34,211,238,0.45); box-shadow: inset 0 0 20px rgba(34,211,238,0.08);
}
.video-item-body { min-width: 0; flex: 1; }
.video-item-name {
  font-size: 12.5px; line-height: 1.45; color: var(--text);
  overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.video-item-meta { margin-top: 5px; font-size: 10.5px; color: var(--faint); font-family: var(--mono); }
.dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; background: var(--faint); }
.dot-done { background: var(--lime); box-shadow: 0 0 10px rgba(163,230,53,0.7); }
.dot-processing { background: var(--amber); animation: pulse 1.2s ease-in-out infinite; }
.dot-failed { background: var(--red); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

/* ---------- 中间视频舞台 ---------- */
.stage { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px; }
.stage-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 2px; }
.stage-title { min-width: 0; display: flex; align-items: center; gap: 10px; }
.stage-name { font-size: 14px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stage-name.dim { color: var(--faint); font-weight: 500; }
.stage-state { font-size: 11px; font-family: var(--mono); color: var(--amber); flex-shrink: 0; }
.stage-state.ok { color: var(--lime); }
.mode-switch { display: flex; gap: 4px; padding: 3px; border-radius: 10px; background: rgba(255,255,255,0.04); border: 1px solid var(--line); flex-shrink: 0; }
.mode-switch button {
  padding: 6px 14px; border: none; border-radius: 8px; background: transparent;
  color: var(--dim); font-size: 12px; cursor: pointer; transition: 0.18s;
}
.mode-switch button.active { background: var(--grad); color: #04121a; font-weight: 700; box-shadow: 0 4px 16px rgba(34,211,238,0.3); }

.player-shell { flex: 1; min-height: 0; display: flex; }
.player-box {
  position: relative; flex: 1; min-height: 0; border-radius: 18px; overflow: hidden;
  background: #000; border: 1px solid var(--line-strong);
  box-shadow: 0 24px 70px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(34,211,238,0.06);
  display: flex; align-items: center; justify-content: center;
  transition: box-shadow 0.4s;
}
.player-box video { width: 100%; height: 100%; object-fit: contain; display: block; background: #000; }
.player-box.seek-flash { box-shadow: 0 0 0 2px var(--cyan), 0 0 60px rgba(34,211,238,0.55); }
.player-empty { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; pointer-events: none; }
.scan {
  position: absolute; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(34,211,238,0.8), transparent);
  animation: scan 3.2s linear infinite; opacity: 0.7;
}
@keyframes scan { 0% { top: 8%; opacity: 0; } 15% { opacity: 0.8; } 85% { opacity: 0.8; } 100% { top: 92%; opacity: 0; } }
.play-orb {
  width: 62px; height: 62px; border-radius: 50%; font-size: 20px; color: var(--cyan);
  border: 1px solid rgba(34,211,238,0.45);
  background: radial-gradient(circle, rgba(34,211,238,0.18), transparent 70%);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 40px rgba(34,211,238,0.35); animation: breathe 3s ease-in-out infinite;
}
.player-empty-text { font-size: 13px; color: var(--faint); }
.hint-bar {
  padding: 9px 14px; border-radius: 11px; font-size: 12.5px;
  color: var(--amber); background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.28);
  animation: fadeUp 0.22s ease-out;
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

/* ---------- 右侧对话 ---------- */
.chat-panel { width: 400px; flex-shrink: 0; }
.chat { flex: 1; overflow-y: auto; padding: 4px 14px 12px; display: flex; flex-direction: column; gap: 12px; }
.chat-empty { margin: auto; text-align: center; color: var(--faint); font-size: 12.5px; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.chat-empty-icon { font-size: 26px; color: rgba(34,211,238,0.5); }
.msg { max-width: 88%; padding: 11px 14px; border-radius: 13px; font-size: 13px; line-height: 1.68; white-space: pre-wrap; animation: fadeUp 0.2s ease-out; }
.msg.user { align-self: flex-end; background: linear-gradient(120deg, rgba(34,211,238,0.2), rgba(139,92,246,0.16)); border: 1px solid rgba(34,211,238,0.4); color: #c9f7ff; border-bottom-right-radius: 4px; }
.msg.bot { align-self: flex-start; background: var(--card); border: 1px solid var(--line); border-bottom-left-radius: 4px; }
.msg-summary {
  margin-bottom: 10px; padding: 10px 12px; border-radius: 10px;
  background: linear-gradient(120deg, rgba(34,211,238,0.13), rgba(139,92,246,0.1));
  border: 1px solid rgba(34,211,238,0.35);
  font-size: 13px; line-height: 1.65; color: #d7f7ff;
}
.summary-label {
  display: inline-block; margin-right: 7px; padding: 1px 8px; border-radius: 999px;
  font-size: 10px; font-weight: 800; letter-spacing: 1px; vertical-align: 1px;
  background: var(--grad); color: #04121a;
}
.summary-text { vertical-align: baseline; }
.chunks { margin-bottom: 10px; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.chunks-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 12px; font-size: 11.5px; color: var(--dim); cursor: pointer;
  background: rgba(255, 255, 255, 0.03); transition: background 0.15s;
}
.chunks-head:hover { background: rgba(255, 255, 255, 0.06); }
.chunks-toggle { font-family: var(--mono); font-size: 10.5px; color: var(--cyan); }
.chunks-body { padding: 6px 12px 10px; display: flex; flex-direction: column; gap: 7px; }
.chunk { display: flex; gap: 8px; font-size: 11.5px; line-height: 1.55; }
.chunk-time { font-family: var(--mono); color: var(--cyan); flex-shrink: 0; font-size: 10.5px; padding-top: 1px; }
.chunk-text { color: var(--dim); }
.msg-title { font-weight: 700; color: var(--cyan); margin-bottom: 6px; }
.msg-concl { margin-top: 2px; }
.msg-critique { margin-top: 8px; font-size: 10.5px; color: var(--faint); font-family: var(--mono); }
.hits { margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--line); }
.hits-label { font-size: 10.5px; color: var(--faint); font-weight: 700; letter-spacing: 0.6px; margin-bottom: 6px; }
.hit { padding: 4px 0; font-size: 12px; color: var(--dim); line-height: 1.55; }
.ts {
  color: var(--cyan); font-family: var(--mono); font-weight: 700; margin-right: 6px; cursor: pointer;
  text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 3px; transition: 0.15s;
}
.ts:hover { color: #fff; text-shadow: 0 0 10px var(--cyan); }
.src { display: inline-block; margin-right: 6px; padding: 1px 7px; border-radius: 999px; font-size: 9.5px; font-family: var(--mono); vertical-align: middle; }
.src.asr { color: var(--cyan); background: rgba(34,211,238,0.13); border: 1px solid rgba(34,211,238,0.28); }
.src.ocr { color: var(--amber); background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.28); }
.agent-progress { padding: 12px 14px; border-radius: 12px; background: var(--card); border: 1px solid var(--line); display: flex; flex-direction: column; gap: 9px; }
.stage-badges { display: flex; flex-wrap: wrap; gap: 6px; }
.stage-badge { font-family: var(--mono); font-size: 10.5px; padding: 3px 9px; border-radius: 7px; border: 1px solid var(--line-strong); color: var(--dim); }
.stage-badge.st-critic { border-color: var(--cyan); color: var(--cyan); }
.stage-badge.st-verify { border-color: var(--violet); color: #b9a3ff; }
.stage-now { display: flex; align-items: center; gap: 9px; font-size: 12.5px; color: var(--dim); }
.spinner { width: 13px; height: 13px; border-radius: 50%; border: 2px solid rgba(120,160,200,0.28); border-top-color: var(--cyan); animation: spin 0.8s linear infinite; flex-shrink: 0; }
@keyframes spin { to { transform: rotate(360deg); } }
.stage-text { color: var(--text); }
.stage-detail { color: var(--faint); font-size: 11.5px; }
.stage-elapsed { margin-left: auto; font-family: var(--mono); font-size: 11.5px; color: var(--faint); }
.input-row { display: flex; gap: 8px; padding: 12px 14px 14px; border-top: 1px solid var(--line); }
.input-row input {
  flex: 1; padding: 12px 15px; border-radius: 11px; font-size: 13.5px; outline: none;
  border: 1px solid var(--line); background: rgba(8,12,22,0.8); color: var(--text); transition: 0.18s;
}
.input-row input:focus { border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(34,211,238,0.14); }
.input-row button {
  padding: 12px 22px; border: none; border-radius: 11px; font-size: 13.5px; font-weight: 800;
  background: var(--grad); color: #04121a; cursor: pointer; transition: 0.2s;
}
.input-row button:hover:not(:disabled) { box-shadow: 0 0 26px rgba(34,211,238,0.5); transform: translateY(-1px); }
.input-row button:disabled { background: rgba(120,160,200,0.12); color: var(--faint); cursor: not-allowed; }

/* ---------- 窄屏：改成上下堆叠 ---------- */
@media (max-width: 1180px) {
  .body { flex-direction: column; overflow-y: auto; }
  .sidebar { width: 100%; max-height: 210px; }
  .chat-panel { width: 100%; min-height: 340px; }
  .stage { min-height: 320px; }
}
</style>
