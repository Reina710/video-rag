// useChat.js —— 问答 + Agent 实时进度（组合式函数）
import { ref } from 'vue'
import { askRag, askAgentStream as askAgentStreamRequest } from './api'
import { readSseStream } from './sse'

const STAGE_LABEL = {
  rewrite: '问题改写',
  retrieval: '检索相关片段',
  planner: 'Planner 拆解任务',
  executor: 'Executor 生成产物',
  verify: '代码核验证据',
  critic: 'Critic 审核',
}

export function useChat() {
  const mode = ref('agent') // 'rag' | 'agent'
  const question = ref('')
  const messages = ref([])
  const loading = ref(false)

  // Agent 实时进度（SSE 事件流驱动）
  const stage = ref('')
  const stageText = ref('')
  const stageDetail = ref('')
  const stages = ref([])
  const elapsed = ref(0)
  let elapsedTimer = null

  function resetStage() {
    stage.value = ''
    stageText.value = ''
    stageDetail.value = ''
    stages.value = []
    elapsed.value = 0
  }

  function startElapsedTimer() {
    resetStage()
    const t0 = Date.now()
    elapsedTimer = setInterval(() => {
      elapsed.value = Math.floor((Date.now() - t0) / 1000)
    }, 1000)
  }

  function stopElapsedTimer() {
    clearInterval(elapsedTimer)
    elapsedTimer = null
  }

  // 处理一个 SSE 阶段事件，更新进度 UI
  function handleStageEvent(ev) {
    const key = ev.stage
    if (ev.status === 'start') {
      stage.value = key
      stageText.value = STAGE_LABEL[key] || key
      stageDetail.value = ev.round ? `第 ${ev.round} 轮` : ''
    } else if (ev.status === 'done') {
      const label = STAGE_LABEL[key] || key
      let detail = ''
      if (key === 'retrieval') detail = `${ev.count || 0} 个片段`
      else if (key === 'planner') detail = `${(ev.tasks || []).length} 个任务`
      else if (key === 'executor') detail = `${ev.conclusions || 0} 结论 / ${ev.evidence || 0} 证据`
      else if (key === 'verify') detail = `${ev.passed}/${ev.total} 通过`
      else if (key === 'critic') detail = ev.passed ? '通过' : '打回重做'
      else if (key === 'rewrite') detail = ev.query || ''
      stages.value.push({ key, label, detail, round: ev.round })
      stage.value = ''
      stageText.value = ''
      stageDetail.value = ''
    }
  }

  // 处理一条 SSE 帧：done / error / 阶段事件 三种
  function handleFrame(ev) {
    if (ev.stage === 'done') {
      const r = ev.result
      messages.value.push({
        role: 'bot',
        summary: r.result.summary || '',
        title: r.result.title,
        conclusions: r.result.conclusions || [],
        evidence: r.result.evidence || [],
        critique: r.critique,
        rounds: r.rounds,
        chunks: r.chunks || [],
      })
    } else if (ev.stage === 'error') {
      messages.value.push({
        role: 'bot',
        text: '出错了：' + (ev.error && (ev.error.detail || ev.error.type)),
      })
    } else {
      handleStageEvent(ev)
    }
  }

  // 把最近 2 轮整理成 [{role, text}]，给后端做追问改写
  function buildHistory() {
    return messages.value.slice(-4).map((m) => ({
      role: m.role,
      text: (m.text || m.summary || m.title || '').slice(0, 200),
    }))
  }

  async function ask(mediaId) {
    const q = question.value.trim()
    if (!q) return
    const history = buildHistory()
    question.value = ''
    messages.value.push({ role: 'user', text: q })
    loading.value = true

    // Agent 模式走 SSE 流式接口，实时展示「Agent 正在做什么」
    if (mode.value === 'agent') {
      await askAgentStream(q, mediaId, history)
      return
    }

    // 普通 RAG 模式：走原单轮接口
    try {
      const data = await askRag(q, mediaId, history)
      if (data.error) {
        messages.value.push({ role: 'bot', text: '出错了：' + data.error })
      } else {
        messages.value.push({ role: 'bot', text: data.answer, evidence: data.hits || [] })
      }
    } catch (err) {
      messages.value.push({ role: 'bot', text: '请求失败：' + err })
    } finally {
      loading.value = false
    }
  }

  async function askAgentStream(q, mediaId, history) {
    startElapsedTimer()
    try {
      const resp = await askAgentStreamRequest(q, mediaId, history)
      await readSseStream(resp, handleFrame)
    } catch (err) {
      // readSseStream 抛的 HTTP 错误带 .data；网络错误是普通 Error
      const text =
        err && err.data !== undefined
          ? '出错了：' + (err.data.error || err.status)
          : '请求失败：' + err
      messages.value.push({ role: 'bot', text })
    } finally {
      stopElapsedTimer()
      loading.value = false
      stage.value = ''
      stageText.value = ''
      stageDetail.value = ''
    }
  }

  function clear() {
    messages.value = []
  }

  return {
    mode,
    question,
    messages,
    loading,
    stage,
    stageText,
    stageDetail,
    stages,
    elapsed,
    ask,
    clear,
  }
}
