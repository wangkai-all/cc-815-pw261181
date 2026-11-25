"""关联快照：两侧表的不可变视图与逐键统计。

快照在构造时刻确定参与关联的键集合与各键有效行数；之后即使两侧
SideData 继续追加新段，已构造的快照行为不变（抽样器只通过快照访问
计数与行号），从而保证「计数和取样来自同一快照」。

空值键（NULL/NaN）不参与内关联：两侧空键行分别记入 null_key_rows，
既不产生关联对，也不进入抽样权重。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterator

from .conditions import Condition
from .store import SideData


@dataclass(frozen=True)
class KeyStat:
    """单个键在快照中的统计（不含条件接受率，接受率在 sampler 中计算）。"""

    key: Any
    left_rows: int
    right_rows: int
    pair_count: int  # = left_rows * right_rows，无条件下的关联对数


class JoinSnapshot:
    """两侧表在某一时刻的内关联快照。"""

    def __init__(self, left: SideData, right: SideData, condition: Condition | None = None):
        if left.key != right.key:
            raise ValueError(
                f"两侧键列名不一致: {left.key!r} vs {right.key!r}"
            )
        self.left = left
        self.right = right
        self.key_name = left.key
        self.condition = condition or Condition.compile(None)

        # 仅两侧都出现的键构成内关联键集合，按稳定顺序排列便于复现。
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        common = left_keys & right_keys
        self._keys: list[Any] = sorted(common, key=_sort_key)
        self._key_set = set(self._keys)

        self._stats: dict[Any, KeyStat] = {}
        total_pairs = 0
        for k in self._keys:
            nl = left.key_count(k)
            nr = right.key_count(k)
            pairs = nl * nr
            self._stats[k] = KeyStat(k, nl, nr, pairs)
            total_pairs += pairs
        self.total_pair_count = total_pairs
        # 快照时刻的表规模在此固化；SideData 之后追加段不会改写这些值。
        self._left_row_count = left.row_count
        self._right_row_count = right.row_count
        self.null_left_rows = left.null_key_rows
        self.null_right_rows = right.null_key_rows
        self._left_distinct = left.distinct_key_count()
        self._right_distinct = right.distinct_key_count()

        h = hashlib.sha256()
        h.update(b"snapshot-v1\n")
        h.update(left.fingerprint_hex().encode())
        h.update(b"\n")
        h.update(right.fingerprint_hex().encode())
        h.update(b"\n")
        h.update(repr(self.condition.expression).encode("utf-8"))
        self._fingerprint = h.hexdigest()

    # -- 键与统计 -------------------------------------------------------
    def keys(self) -> list[Any]:
        return list(self._keys)

    def has_key(self, key: Any) -> bool:
        return key in self._key_set

    def stat(self, key: Any) -> KeyStat:
        return self._stats[key]

    def iter_stats(self) -> Iterator[KeyStat]:
        return (self._stats[k] for k in self._keys)

    def left_row_ids(self, key: Any) -> list[int]:
        return list(self.left.key_rows(key))

    def right_row_ids(self, key: Any) -> list[int]:
        return list(self.right.key_rows(key))

    def get_left_row(self, row_id: int) -> dict:
        return self.left.get_row(row_id)

    def get_right_row(self, row_id: int) -> dict:
        return self.right.get_row(row_id)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def describe(self) -> dict[str, Any]:
        return {
            "key_name": self.key_name,
            "condition": self.condition.expression,
            "left_rows": self._left_row_count,
            "right_rows": self._right_row_count,
            "left_null_key_rows": self.null_left_rows,
            "right_null_key_rows": self.null_right_rows,
            "left_distinct_keys": self._left_distinct,
            "right_distinct_keys": self._right_distinct,
            "join_key_count": len(self._keys),
            "total_pair_count": self.total_pair_count,
            "fingerprint": self._fingerprint,
        }


# 可比较标量的固定类型序；None/bool 与 int 同为数字族但标签不同，
# 避免 Python 里 True == 1 的哈希/比较混淆影响排序稳定性。
_SCALAR_RANK = {
    type(None): 0,
    bool: 1,
    int: 2,
    float: 3,
    str: 4,
}


def _sort_key(k: Any):
    """异构键（含异构元素 tuple）的全序排序键，绝不抛 TypeError。

    每个标量都包成 ``(类型秩, 值)``，tuple 递归打包，保证比较双方在
    每一位置上类型标签一致；无法识别的类型回退到其 repr 字符串。
    注意：Python 语义下 ``1 == True``、``1.0 == 1`` 哈希同桶，这是语言
    既定行为，排序键只负责给出确定性顺序，不改变字典归并语义。
    """
    rank = _SCALAR_RANK.get(type(k))
    if rank is not None:
        return (rank, k)
    if isinstance(k, tuple):
        return (5, tuple(_sort_key(item) for item in k))
    # 其余可哈希/不可哈希的杂类一律字符串化，保证总可比较。
    return (6, repr(k))


def build_snapshot(
    left: SideData,
    right: SideData,
    condition: str | Condition | None = None,
) -> JoinSnapshot:
    cond = condition if isinstance(condition, Condition) else Condition.compile(condition)
    return JoinSnapshot(left, right, cond)
