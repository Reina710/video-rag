// sse.js —— 解析 SSE（Server-Sent Events）流。
// 纯函数：不依赖 Vue，输入 fetch 的 Response，逐帧回调，方便单测。

export async function readSseStream(response, onEvent) {
  // 非 2xx 或没有 body：把错误解析出来抛给调用方（带 data 字段便于区分 HTTP 错误）
  if (!response.ok || !response.body) {
    let data = {}
    try {
      data = await response.json()
    } catch (e) {
      /* 忽略：body 可能不是 JSON */
    }
    const err = new Error(data.error || ('HTTP ' + response.status))
    err.data = data
    err.status = response.status
    throw err
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE 帧以 \n\n 分隔，逐帧解析
      let idx
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const raw = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        const line = raw.trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        let ev
        try {
          ev = JSON.parse(payload)
        } catch (e) {
          continue // 非法 JSON 帧直接跳过
        }
        onEvent(ev)
      }
    }
  } finally {
    try {
      reader.releaseLock()
    } catch (e) {
      /* 忽略 */
    }
  }
}
