// format.test.js —— 时间格式化纯函数测试
import { describe, it, expect } from 'vitest'
import { fmtMs, formatElapsed } from './format'

describe('fmtMs（毫秒 → mm:ss）', () => {
  it('0 毫秒 → 00:00', () => {
    expect(fmtMs(0)).toBe('00:00')
  })

  it('90000 毫秒 → 01:30', () => {
    expect(fmtMs(90000)).toBe('01:30')
  })

  it('60000 毫秒 → 01:00', () => {
    expect(fmtMs(60000)).toBe('01:00')
  })

  it('3661000 毫秒 → 61:01（超过 1 小时不额外进位）', () => {
    expect(fmtMs(3661000)).toBe('61:01')
  })

  it('不足 10 秒时补前导零', () => {
    expect(fmtMs(5000)).toBe('00:05')
  })
})

describe('formatElapsed（秒 → mm:ss）', () => {
  it('0 秒 → 00:00', () => {
    expect(formatElapsed(0)).toBe('00:00')
  })

  it('65 秒 → 01:05', () => {
    expect(formatElapsed(65)).toBe('01:05')
  })

  it('3600 秒 → 60:00', () => {
    expect(formatElapsed(3600)).toBe('60:00')
  })
})
