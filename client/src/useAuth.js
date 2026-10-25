// useAuth.js —— 登录/注册/登出状态与逻辑（组合式函数）
import { ref } from 'vue'
import { fetchMe, login, register, logout as logoutRequest } from './api'

export function useAuth() {
  const loggedIn = ref(false)
  const username = ref('')
  const authTab = ref('login') // 'login' | 'register'
  const authUsername = ref('')
  const authPassword = ref('')
  const authPassword2 = ref('')
  const authError = ref('')
  const authSuccess = ref('')

  // 刷新登录态，返回是否已登录（调用方决定要不要联动加载视频列表）
  async function refreshAuth() {
    try {
      const d = await fetchMe()
      loggedIn.value = d.loggedIn
      username.value = d.username || ''
      return d.loggedIn
    } catch (e) {
      return false
    }
  }

  function switchTab(tab) {
    authTab.value = tab
    authError.value = ''
    authSuccess.value = ''
    authPassword.value = ''
    authPassword2.value = ''
  }

  // 提交登录/注册，返回是否成功（登录成功后内部已刷新登录态）
  async function submitAuth() {
    authError.value = ''
    authSuccess.value = ''
    const u = authUsername.value.trim()
    const p = authPassword.value
    if (!u || !p) {
      authError.value = '请输入用户名和密码'
      return false
    }

    if (authTab.value === 'register') {
      if (p.length < 6) {
        authError.value = '密码至少 6 位'
        return false
      }
      if (p !== authPassword2.value) {
        authError.value = '两次输入的密码不一致'
        return false
      }
    }

    try {
      if (authTab.value === 'register') {
        const r = await register(u, p)
        if (!r.ok) {
          const d = await r.json().catch(() => ({}))
          authError.value = d.error || '注册失败，请重试'
          return false
        }
        authSuccess.value = '注册成功，请登录'
        authTab.value = 'login'
        authPassword.value = ''
        authPassword2.value = ''
      } else {
        const r = await login(u, p)
        if (!r.ok) {
          const d = await r.json().catch(() => ({}))
          authError.value = d.error || '用户名或密码错误'
          return false
        }
        authPassword.value = ''
        authPassword2.value = ''
        await refreshAuth()
      }
      return true
    } catch (e) {
      authError.value = '网络请求失败，请检查后端服务是否启动'
      return false
    }
  }

  async function logout() {
    // 本地状态无条件清空；后端清 session 失败了也照清（下次 /me 自然返回未登录）
    try {
      await logoutRequest()
    } catch (e) {
      /* 忽略：登出不应被后端状态阻塞 */
    } finally {
      loggedIn.value = false
      username.value = ''
    }
  }

  return {
    loggedIn,
    username,
    authTab,
    authUsername,
    authPassword,
    authPassword2,
    authError,
    authSuccess,
    refreshAuth,
    switchTab,
    submitAuth,
    logout,
  }
}
