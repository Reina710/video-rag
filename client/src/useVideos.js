// useVideos.js —— 视频列表状态（组合式函数）
import { ref } from 'vue'
import { fetchVideos, fetchVideoStatus } from './api'
import { uploadFileInChunks } from './chunkUpload'

export function useVideos() {
  const videos = ref([])

  async function loadVideos() {
    try {
      const d = await fetchVideos()
      videos.value = d.videos || []
    } catch (e) {
      /* 忽略：后端没起时静默，页面不崩 */
    }
  }

  // 分片上传：onProgress(done, total) 给 UI 画进度条
  async function upload(file, onProgress) {
    try {
      return await uploadFileInChunks(file, { onProgress })
    } catch (e) {
      return { error: (e && e.message) || '上传失败' }
    }
  }

  // 轮询解析进度：status 变成 done/failed 就停
  async function pollStatus(videoId, onUpdate, intervalMs = 2000, timeoutMs = 20 * 60 * 1000) {
    const start = Date.now()
    while (Date.now() - start < timeoutMs) {
      try {
        const d = await fetchVideoStatus(videoId)
        const v = d.video
        if (v) {
          if (onUpdate) onUpdate(v)
          if (v.status === 'done' || v.status === 'failed') {
            loadVideos()
            return v
          }
        }
      } catch (e) {
        /* 网络抖动继续轮询 */
      }
      await new Promise((r) => setTimeout(r, intervalMs))
    }
    return null
  }

  function clear() {
    videos.value = []
  }

  return { videos, loadVideos, upload, pollStatus, clear }
}
