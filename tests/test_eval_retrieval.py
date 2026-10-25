# -*- coding: utf-8 -*-
"""评测脚本的纯函数测试：时间区间重叠判断。

这个函数决定 hit@K 算不算命中，改错一个 < 就会让指标全错，最值得测。
跑：python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_retrieval import _overlaps


class TestOverlaps(unittest.TestCase):
    def test_完全包含算命中(self):
        self.assertTrue(_overlaps(10000, 20000, 5000, 30000))

    def test_部分重叠算命中(self):
        self.assertTrue(_overlaps(8000, 15000, 12000, 20000))

    def test_边界相接不算命中(self):
        # [0,1000) 和 [1000,2000) 是相邻但不重叠
        self.assertFalse(_overlaps(0, 1000, 1000, 2000))

    def test_完全错开不算命中(self):
        self.assertFalse(_overlaps(0, 1000, 5000, 6000))

    def test_零长度区间(self):
        self.assertFalse(_overlaps(1000, 1000, 2000, 3000))


if __name__ == "__main__":
    unittest.main()
