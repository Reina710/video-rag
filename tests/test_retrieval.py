# -*- coding: utf-8 -*-
"""检索层单元测试：只测纯函数 + 可注入的语料，不依赖网络 / Qdrant。

为什么测这几块：
    检索里最容易"改一个参数就悄悄坏掉"的是排序和合并逻辑，而且它们都是纯函数，
    最好测。扩窗就出过事 —— 合并时按时间排序丢了相关性排名，hit@1 从 87.5% 掉到 37.5%，
    有这条用例就能立刻发现。

跑：python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import retrieval


class TestRRF(unittest.TestCase):
    def test_两路都排第一的应该胜出(self):
        a = [{"key": "x"}, {"key": "y"}]
        b = [{"key": "x"}, {"key": "z"}]
        fused = retrieval.rrf_fuse([a, b])
        self.assertEqual(fused[0]["key"], "x")

    def test_只看排名不看分数(self):
        # RRF 不该读 score 字段：这里故意不带 score，也不该报错
        a = [{"key": "a"}, {"key": "b"}]
        b = [{"key": "b"}, {"key": "a"}]
        fused = retrieval.rrf_fuse([a, b], k=60)
        self.assertEqual(sorted(d["key"] for d in fused), ["a", "b"])

    def test_空输入返回空(self):
        self.assertEqual(retrieval.rrf_fuse([]), [])
        self.assertEqual(retrieval.rrf_fuse([[], []]), [])

    def test_rrf分数单调(self):
        a = [{"key": "a"}, {"key": "b"}, {"key": "c"}]
        fused = retrieval.rrf_fuse([a], k=60)
        scores = [d["rrf"] for d in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestExpandWindows(unittest.TestCase):
    """用假语料测扩窗，不碰真实 Qdrant。"""

    def setUp(self):
        docs = []
        for i, text in enumerate(["本季最强的", "就是它有星星", "碧蓝之海", "超版女友"]):
            docs.append({
                "key": "k%d" % i,
                "start": i * 2000,
                "end": i * 2000 + 2000,
                "source": "asr",
                "text": text,
                "media_id": 5,
                "tokens": [],
            })
        self._orig = retrieval._corpus
        retrieval._corpus = lambda: {"docs": docs}

    def tearDown(self):
        retrieval._corpus = self._orig

    def test_命中后把相邻句一起带出来(self):
        # 这就是"本季最强的"和"碧蓝之海"被切断的真实场景
        hit = {"start": 0, "end": 2000, "source": "asr", "text": "本季最强的", "media_id": 5}
        out = retrieval.expand_windows([hit], before_ms=2000, after_ms=8000)
        self.assertEqual(len(out), 1)
        self.assertIn("碧蓝之海", out[0]["text"])

    def test_扩窗后窗口起点早于锚点_对外必须用anchor(self):
        """窗口往前补上下文，start 会比 anchor 早；展示和跳转只能用 anchor。

        /ask 接口曾经直接取 start 对外，结果时间戳比真实命中位置早 8 秒，
        点过去听到的是上一句。这条用例锁住"两者不相等"这个前提。
        """
        hit = {"start": 4000, "end": 6000, "source": "asr", "text": "碧蓝之海", "media_id": 5}
        out = retrieval.expand_windows([hit])[0]
        self.assertEqual(out["anchor"], 4000)
        self.assertLess(out["start"], out["anchor"])

    def test_合并后按原排名排序而不是按时间(self):
        # 第 0 名命中在 6 秒，第 1 名命中在 0 秒。
        # 合并后必须还是 6 秒那条在前（相关性优先）。
        hits = [
            {"start": 6000, "end": 8000, "source": "asr", "text": "碧蓝之海", "media_id": 5},
            {"start": 0, "end": 2000, "source": "asr", "text": "本季最强的", "media_id": 5},
        ]
        out = retrieval.expand_windows(hits, before_ms=0, after_ms=0)
        self.assertEqual(out[0]["start"], 6000)

    def test_时间不重叠不合并(self):
        hits = [
            {"start": 0, "end": 2000, "source": "asr", "text": "a", "media_id": 5},
            {"start": 6000, "end": 8000, "source": "asr", "text": "b", "media_id": 5},
        ]
        self.assertEqual(len(retrieval.expand_windows(hits, before_ms=0, after_ms=0)), 2)

    def test_不同来源不合并(self):
        # ASR 和 OCR 是两条线，混在一起会把语音和画面文字粘成一坨
        hits = [
            {"start": 0, "end": 2000, "source": "asr", "text": "语音", "media_id": 5},
            {"start": 1000, "end": 3000, "source": "ocr", "text": "画面", "media_id": 5},
        ]
        self.assertEqual(len(retrieval.expand_windows(hits, before_ms=0, after_ms=0)), 2)

    def test_空输入返回空(self):
        self.assertEqual(retrieval.expand_windows([]), [])


class TestExpandWindowsOCR(unittest.TestCase):
    """回归用例：OCR 帧首尾相接时不能链式合并（曾出现"点过去对不上"）。"""

    def setUp(self):
        # 模拟 OCR：每 30 秒一帧、每帧覆盖 30 秒（真实数据就是这样）
        docs = []
        for i in range(6):
            docs.append({
                "key": "o%d" % i,
                "start": i * 30000,
                "end": (i + 1) * 30000,
                "source": "ocr",
                "text": "帧%d" % i,
                "media_id": 5,
                "tokens": [],
            })
        self._orig = retrieval._corpus
        retrieval._corpus = lambda: {"docs": docs}

    def tearDown(self):
        retrieval._corpus = self._orig

    def test_命中90秒的帧_窗口不能吞到30秒(self):
        hit = {"start": 90000, "end": 120000, "source": "ocr", "text": "帧3", "media_id": 5}
        out = retrieval.expand_windows([hit])
        self.assertEqual(len(out), 1)
        # 修 bug 前这里会退到 30000（显示 00:30），但内容其实在 01:30
        self.assertGreaterEqual(out[0]["start"], 60000)
        # 锚点必须还是命中那一帧
        self.assertEqual(out[0]["anchor"], 90000)

    def test_锚点差30秒的两帧不合并(self):
        hits = [
            {"start": 90000, "end": 120000, "source": "ocr", "text": "帧3", "media_id": 5},
            {"start": 120000, "end": 150000, "source": "ocr", "text": "帧4", "media_id": 5},
        ]
        out = retrieval.expand_windows(hits)
        self.assertEqual(len(out), 2)
        self.assertEqual({w["anchor"] for w in out}, {90000, 120000})

    def test_锚点很近的才合并(self):
        # 同一帧附近的两条命中（ASR 那种 2~3 秒一句）应该合成一条
        docs = [
            {"key": "a0", "start": 1000, "end": 3000, "source": "asr", "text": "本季最强的", "media_id": 5, "tokens": []},
            {"key": "a1", "start": 3000, "end": 5000, "source": "asr", "text": "碧蓝之海", "media_id": 5, "tokens": []},
        ]
        orig = retrieval._corpus
        retrieval._corpus = lambda: {"docs": docs}
        try:
            hits = [
                {"start": 1000, "end": 3000, "source": "asr", "text": "本季最强的", "media_id": 5},
                {"start": 3000, "end": 5000, "source": "asr", "text": "碧蓝之海", "media_id": 5},
            ]
            out = retrieval.expand_windows(hits)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["anchor"], 1000)
            self.assertIn("碧蓝之海", out[0]["text"])
        finally:
            retrieval._corpus = orig


class TestTokenize(unittest.TestCase):
    def test_中文分词(self):
        toks = retrieval._tokenize("本季最强的作品")
        self.assertTrue(any("本季" in t for t in toks))

    def test_停用词被过滤(self):
        self.assertEqual(retrieval._tokenize("的了"), [])

    def test_纯标点被过滤(self):
        self.assertEqual(retrieval._tokenize("，。！？"), [])

    def test_繁体转简体后能对上(self):
        # ASR 输出繁体、用户提问简体。不归一的话「漫畫」和「漫画」是两个词，
        # BM25 一条都召不回来，融合反而拉低 Top-1。
        trad = set(retrieval._tokenize("我最愛的漫畫"))
        simp = set(retrieval._tokenize("我最爱的漫画"))
        self.assertTrue(trad & simp, "繁简归一失败：%s vs %s" % (trad, simp))

    def test_繁简归一不只影响查询(self):
        self.assertIn("漫画", retrieval._tokenize("漫畫"))


if __name__ == "__main__":
    unittest.main()
