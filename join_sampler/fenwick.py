"""Fenwick 树（二叉索引树）：按权重抽键与无放回扣权。

两个变体：
* IntFenwick：权重为整数（无条件 / 精确接受率），抽键后按整数扣减；
* FloatFenwick：权重为浮点（估计接受率校正），容忍浮点累计误差。

抽键采用均匀 u ∈ [1, total] 后按前缀和定位，时间 O(log n)；
无放回模式抽中某键后只扣该键权重，其余键权重不动，保证任意存活对
被抽中的条件概率始终相等（键权重 ∝ 该键存活可接受对数）。
"""

from __future__ import annotations

import random
from typing import Sequence


class IntFenwick:
    """整数权重 Fenwick 树，下标 1..n。"""

    __slots__ = ("n", "_tree", "_total")

    def __init__(self, weights: Sequence[int]) -> None:
        self.n = len(weights)
        self._tree = [0] * (self.n + 1)
        total = 0
        for i, w in enumerate(weights, start=1):
            self._tree[i] += w
            parent = i + (i & -i)
            if parent <= self.n:
                self._tree[parent] += self._tree[i]
            total += w
        self._total = total

    @property
    def total(self) -> int:
        return self._total

    def weight(self, index0: int) -> int:
        """返回当前单点权重（由前缀和差分得到）。"""
        idx = index0 + 1
        value = self._tree[idx]
        low = idx - (idx & -idx)
        j = idx - 1
        while j > low:
            value -= self._tree[j]
            j -= j & -j
        return value

    def add(self, index0: int, delta: int) -> None:
        i = index0 + 1
        while i <= self.n:
            self._tree[i] += delta
            i += i & -i
        self._total += delta

    def pick(self, rng: random.Random) -> int:
        """按权重返回一个 0 基下标；总权重为 0 时抛出 IndexError。"""
        if self._total <= 0:
            raise IndexError("总权重已为 0，无键可抽")
        target = rng.randint(1, self._total)
        i = 0
        bit = 1 << (self.n.bit_length() - 1)
        while bit:
            nxt = i + bit
            if nxt <= self.n and self._tree[nxt] < target:
                i = nxt
                target -= self._tree[nxt]
            bit >>= 1
        return i  # 0 基下标


class FloatFenwick:
    """浮点权重 Fenwick 树，用于接受率为估计值的校正权重。"""

    __slots__ = ("n", "_tree", "_total", "_eps0")

    def __init__(self, weights: Sequence[float]) -> None:
        self.n = len(weights)
        self._tree = [0.0] * (self.n + 1)
        total = 0.0
        for i, w in enumerate(weights, start=1):
            self._tree[i] += w
            parent = i + (i & -i)
            if parent <= self.n:
                self._tree[parent] += self._tree[i]
            total += w
        self._total = total
        # 相对误差尺度按建树时的总量固定；pick 时取它与当前总量的较大值，
        # 避免总量缩到很小后仍用陈旧的大 epsilon 误杀合法幸存键。
        self._eps0 = 1e-12 * max(1.0, abs(total))

    @property
    def total(self) -> float:
        return self._total

    def _current_eps(self) -> float:
        return max(self._eps0, 1e-12 * max(1.0, abs(self._total)))

    def weight(self, index0: int) -> float:
        """返回当前单点权重（由前缀和差分得到）。"""
        idx = index0 + 1
        value = self._tree[idx]
        low = idx - (idx & -idx)
        j = idx - 1
        while j > low:
            value -= self._tree[j]
            j -= j & -j
        return value

    def add(self, index0: int, delta: float) -> None:
        i = index0 + 1
        while i <= self.n:
            self._tree[i] += delta
            i += i & -i
        self._total += delta

    def pick(self, rng: random.Random) -> int:
        eps = self._current_eps()
        if self._total <= eps:
            raise IndexError("总权重已为 0，无键可抽")
        target = rng.uniform(0.0, self._total)
        i = 0
        bit = 1 << (self.n.bit_length() - 1)
        while bit:
            nxt = i + bit
            if nxt <= self.n and self._tree[nxt] <= target:
                i = nxt
                target -= self._tree[nxt]
            bit >>= 1
        idx = i  # 落在 [0, n-1]
        return min(idx, self.n - 1)
