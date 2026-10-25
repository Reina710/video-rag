// sse.test.js —— 用假 ReadableStream 验证 SSE 流解析逻辑
import { describe, it, expect, vi } from 'vitest'
import { readSseStream } from './sse'

function fakeResponse(bodyStr) {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(bodyStr))
      controller.close()
    },
  })
  return { ok: true, body: stream, status: 200, json: vi.fn() }
}

function splitResponse(parts) {
  const stream = new ReadableStream({
    start(controller) {
      for (const p of parts) {
        controller.enqueue(new TextEncoder().encode(p))
      }
      controller.close()
    },
  })
  return { ok: true, body: stream, status: 200, json: vi.fn() }
}

describe('readSseStream', () => {
  it('解析 data: JSON 帧，逐帧回调', async () => {
    const frames = 'data: {"stage":"planner","status":"start"}\n\ndata: {"stage":"done"}\n\n'
    const events = []
    await readSseStream(fakeResponse(frames), (ev) => events.push(ev))
    expect(events).toEqual([
      { stage: 'planner', status: 'start' },
      { stage: 'done' },
    ])
  })

  it('忽略非 data: 行（注释/心跳）', async () => {
    const frames = ': keepalive\n\ndata: {"stage":"critic","passed":true}\n\n'
    const events = []
    await readSseStream(fakeResponse(frames), (ev) => events.push(ev))
    expect(events).toEqual([{ stage: 'critic', passed: true }])
  })

  it('忽略非法 JSON 帧，不中断后续解析', async () => {
    const frames = 'data: 不是JSON\n\ndata: {"stage":"done"}\n\n'
    const events = []
    await readSseStream(fakeResponse(frames), (ev) => events.push(ev))
    expect(events).toEqual([{ stage: 'done' }])
  })

  it('一帧跨两个 chunk 也能正确拼接解析', async () => {
    const resp = splitResponse([
      'data: {"stage":"planner"',
      '}\n\ndata: {"stage":"done"}\n\n',
    ])
    const events = []
    await readSseStream(resp, (ev) => events.push(ev))
    expect(events).toEqual([{ stage: 'planner' }, { stage: 'done' }])
  })

  it('非 2xx 响应抛错并带 data 字段', async () => {
    const resp = { ok: false, body: null, status: 500, json: vi.fn().mockResolvedValue({ error: '内部错误' }) }
    await expect(readSseStream(resp, () => {})).rejects.toMatchObject({
      status: 500,
      data: { error: '内部错误' },
    })
  })

  it('body 为空且非 ok 时，用 HTTP 状态兜底报错', async () => {
    const resp = { ok: false, body: null, status: 404, json: vi.fn().mockResolvedValue({}) }
    await expect(readSseStream(resp, () => {})).rejects.toMatchObject({ status: 404 })
  })
})
