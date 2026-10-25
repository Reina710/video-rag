// format.js —— 时间格式化纯函数

// 毫秒 -> mm:ss（如 90000 -> "01:30"）
export function fmtMs(ms) {
  const sec = Math.floor(ms / 1000)
  const mm = String(Math.floor(sec / 60)).padStart(2, '0')
  const ss = String(sec % 60).padStart(2, '0')
  return mm + ':' + ss
}

// 秒 -> mm:ss（计时器用，如 65 -> "01:05"）
export function formatElapsed(totalSeconds) {
  const mm = String(Math.floor(totalSeconds / 60)).padStart(2, '0')
  const ss = String(totalSeconds % 60).padStart(2, '0')
  return mm + ':' + ss
}
