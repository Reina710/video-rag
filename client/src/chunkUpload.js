// chunkUpload.js —— 分片上传 + 断点续传。
//
// 为什么分片：大视频一次性 POST 容易在网关/浏览器超时，断了要从头上传；
// 切成 5MB 一片后，断了只需要补缺的那几片（先查 /api/upload/status）。
//
// 面试可讲：
//   - 分片大小 5MB 是折中：太小请求数多、太大失败重传代价高
//   - 每片单独重试（指数退避），单片失败不毁掉整个上传
//   - 合并放在服务端做，前端只管把片传齐
import { initUpload, uploadChunk, getUploadStatus, completeUpload } from './api'

export const CHUNK_SIZE = 5 * 1024 * 1024

export async function uploadFileInChunks(file, { onProgress } = {}) {
  const { uploadId } = await initUpload(file.name)
  if (!uploadId) throw new Error('初始化上传失败')

  const total = Math.max(1, Math.ceil(file.size / CHUNK_SIZE))

  // 断点续传：先问后端已经收到了哪些分片
  let uploaded = new Set()
  try {
    const s = await getUploadStatus(uploadId)
    uploaded = new Set(s.uploaded || [])
  } catch (e) {
    // 拿不到状态就从头上传，不影响正确性
  }

  for (let i = 0; i < total; i++) {
    if (uploaded.has(i)) {
      if (onProgress) onProgress(i + 1, total)
      continue
    }
    const blob = file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE)
    await _uploadWithRetry(uploadId, i, blob)
    if (onProgress) onProgress(i + 1, total)
  }

  return completeUpload(uploadId)
}

async function _uploadWithRetry(uploadId, index, blob, attempts = 3) {
  let lastErr
  for (let i = 0; i < attempts; i++) {
    try {
      return await uploadChunk(uploadId, index, blob)
    } catch (e) {
      lastErr = e
      // 指数退避：0.5s / 1s / 2s
      await new Promise((r) => setTimeout(r, 500 * 2 ** i))
    }
  }
  throw lastErr
}
