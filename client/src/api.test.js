// api.test.js —— 用 mock fetch 验证每个接口的 URL、method、body 是否正确
import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  fetchMe, login, register, logout,
  fetchVideos, uploadVideo, askRag, askAgentStream,
} from './api'

describe('api.js', () => {
  beforeEach(() => {
    global.fetch = vi.fn()
  })

  it('login 发 POST /api/login，带 JSON body', async () => {
    global.fetch.mockResolvedValue({ ok: true })
    await login('zhang', '123456')
    expect(global.fetch).toHaveBeenCalledWith('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'zhang', password: '123456' }),
    })
  })

  it('register 发 POST /api/register，带 JSON body', async () => {
    global.fetch.mockResolvedValue({ ok: true })
    await register('zhang', '123456')
    expect(global.fetch).toHaveBeenCalledWith('/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'zhang', password: '123456' }),
    })
  })

  it('fetchMe 发 GET /api/me 并解析 json', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({ loggedIn: true, username: 'zhang' }) })
    const d = await fetchMe()
    expect(global.fetch).toHaveBeenCalledWith('/api/me')
    expect(d).toEqual({ loggedIn: true, username: 'zhang' })
  })

  it('logout 发 POST /api/logout', async () => {
    global.fetch.mockResolvedValue({ ok: true })
    await logout()
    expect(global.fetch).toHaveBeenCalledWith('/api/logout', { method: 'POST' })
  })

  it('fetchVideos 发 GET /api/videos 并解析 json', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({ videos: [] }) })
    const d = await fetchVideos()
    expect(global.fetch).toHaveBeenCalledWith('/api/videos')
    expect(d).toEqual({ videos: [] })
  })

  it('uploadVideo 用 FormData 发 POST /api/upload', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({ filename: 'a.mp4' }) })
    const d = await uploadVideo('dummy-file')
    expect(d).toEqual({ filename: 'a.mp4' })
    const [url, options] = global.fetch.mock.calls[0]
    expect(url).toBe('/api/upload')
    expect(options.method).toBe('POST')
    expect(options.body).toBeInstanceOf(FormData)
  })

  it('askRag 发 POST /api/ask，带 question', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({ answer: 'xx' }) })
    await askRag('视频讲了什么')
    expect(global.fetch).toHaveBeenCalledWith('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: '视频讲了什么' }),
    })
  })

  it('askAgentStream 返回原始 Response（不解析，留给 sse.js）', async () => {
    const resp = { ok: true, body: {} }
    global.fetch.mockResolvedValue(resp)
    const r = await askAgentStream('问题')
    expect(r).toBe(resp)
  })
})
