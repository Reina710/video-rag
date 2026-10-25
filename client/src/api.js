// api.js —— 所有后端 HTTP 调用集中在这一层。
// 目的：① 组件里不再散落 fetch；② 测试时可以整体 mock 这一层。
// 约定：返回「数据」的接口直接 resolve 解析后的 JSON；
//       需要调用方看 HTTP 状态的（login/register/logout/SSE）返回原始 Response。

export function fetchMe() {
  return fetch('/api/me').then((r) => r.json())
}

export function login(username, password) {
  return fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
}

export function register(username, password) {
  return fetch('/api/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
}

export function logout() {
  return fetch('/api/logout', { method: 'POST' })
}

export function fetchVideos() {
  return fetch('/api/videos').then((r) => r.json())
}

export function uploadVideo(file) {
  const fd = new FormData()
  fd.append('file', file)
  return fetch('/api/upload', { method: 'POST', body: fd }).then((r) => r.json())
}

// ---------- 分片上传（大文件用） ----------
export function initUpload(filename) {
  return fetch('/api/upload/init', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename }),
  }).then((r) => r.json())
}

export function uploadChunk(uploadId, index, blob) {
  const fd = new FormData()
  fd.append('uploadId', uploadId)
  fd.append('index', String(index))
  fd.append('chunk', blob)
  return fetch('/api/upload/chunk', { method: 'POST', body: fd }).then((r) => {
    if (!r.ok) throw new Error('分片上传失败: ' + r.status)
    return r.json()
  })
}

export function getUploadStatus(uploadId) {
  return fetch('/api/upload/status?uploadId=' + encodeURIComponent(uploadId)).then((r) => r.json())
}

export function completeUpload(uploadId) {
  return fetch('/api/upload/complete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uploadId }),
  }).then((r) => r.json())
}

export function fetchVideoStatus(id) {
  return fetch('/api/video-status?id=' + id).then((r) => r.json())
}

export function askRag(question, mediaId, history) {
  return fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, mediaId, history }),
  }).then((r) => r.json())
}

// SSE 接口：返回原始 Response，交给 sse.js 的 readSseStream 去解析
export function askAgentStream(question, mediaId, history) {
  return fetch('/api/ask_agent_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, mediaId, history }),
  })
}
