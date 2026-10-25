// useAuth.test.js —— mock api 层，验证登录/注册的校验与状态流转
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useAuth } from './useAuth'

vi.mock('./api', () => ({
  fetchMe: vi.fn(),
  login: vi.fn(),
  register: vi.fn(),
  logout: vi.fn(),
}))

import * as api from './api'

describe('useAuth', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('注册：密码少于 6 位被拦，不发请求', async () => {
    const auth = useAuth()
    auth.authTab.value = 'register'
    auth.authUsername.value = 'zhang'
    auth.authPassword.value = '123'
    auth.authPassword2.value = '123'
    const ok = await auth.submitAuth()
    expect(ok).toBe(false)
    expect(auth.authError.value).toBe('密码至少 6 位')
    expect(api.register).not.toHaveBeenCalled()
  })

  it('注册：两次密码不一致被拦', async () => {
    const auth = useAuth()
    auth.authTab.value = 'register'
    auth.authUsername.value = 'zhang'
    auth.authPassword.value = '123456'
    auth.authPassword2.value = '654321'
    const ok = await auth.submitAuth()
    expect(ok).toBe(false)
    expect(auth.authError.value).toBe('两次输入的密码不一致')
    expect(api.register).not.toHaveBeenCalled()
  })

  it('注册：空用户名/密码被拦', async () => {
    const auth = useAuth()
    auth.authTab.value = 'register'
    auth.authUsername.value = ''
    auth.authPassword.value = ''
    const ok = await auth.submitAuth()
    expect(ok).toBe(false)
    expect(auth.authError.value).toBe('请输入用户名和密码')
  })

  it('登录成功：刷新登录态，返回 true', async () => {
    api.login.mockResolvedValue({ ok: true })
    api.fetchMe.mockResolvedValue({ loggedIn: true, username: 'zhang' })
    const auth = useAuth()
    auth.authTab.value = 'login'
    auth.authUsername.value = 'zhang'
    auth.authPassword.value = '123456'
    const ok = await auth.submitAuth()
    expect(ok).toBe(true)
    expect(auth.loggedIn.value).toBe(true)
    expect(auth.username.value).toBe('zhang')
  })

  it('登录失败：返回后端错误信息', async () => {
    api.login.mockResolvedValue({ ok: false, json: () => Promise.resolve({ error: '用户名或密码错误' }) })
    const auth = useAuth()
    auth.authTab.value = 'login'
    auth.authUsername.value = 'zhang'
    auth.authPassword.value = 'wrong'
    const ok = await auth.submitAuth()
    expect(ok).toBe(false)
    expect(auth.authError.value).toBe('用户名或密码错误')
  })

  it('登出：清除登录态', async () => {
    api.logout.mockResolvedValue({ ok: true })
    const auth = useAuth()
    auth.loggedIn.value = true
    auth.username.value = 'zhang'
    await auth.logout()
    expect(auth.loggedIn.value).toBe(false)
    expect(auth.username.value).toBe('')
  })
})
