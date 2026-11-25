"""分段列式存储与单侧表（SideData）。

为什么需要自己的存储：
* 数据按 *段（segment）* 追加，段一旦追加即不可变；抽样器持有的快照
  在任何后续追加下保持稳定（计数与取样同一快照）；
* 列式 + 段结构让 get_row 只物化被抽中的行，关联结果从不整体物化；
* 即使源表中存在取值完全相同的重复行，每一行仍拥有独立的扁平行号
  （flat row id），因此在有放回模式下它们是彼此独立的行身份。
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


class SegmentedColumn:
    """按段追加的单列存储，支持按扁平下标 O(段数) 取值。"""

    __slots__ = ("_segments", "_length")

    def __init__(self) -> None:
        self._segments: list[list[Any]] = []
        self._length = 0

    def append_segment(self, values: list[Any]) -> None:
        # 复制一份，调用方之后修改原列表不会影响已固化的段。
        self._segments.append(list(values))
        self._length += len(values)

    def __len__(self) -> int:
        return self._length

    def segment_count(self) -> int:
        return len(self._segments)

    def __getitem__(self, flat_index: int) -> Any:
        if flat_index < 0:
            flat_index += self._length
        if not 0 <= flat_index < self._length:
            raise IndexError(f"行号越界: {flat_index} (共 {self._length} 行)")
        # 段数通常很少（每批数据一段），线性定位即可，避免额外偏移表。
        for segment in self._segments:
            if flat_index < len(segment):
                return segment[flat_index]
            flat_index -= len(segment)
        raise IndexError("分段定位失败")  # pragma: no cover

    def iter_values(self) -> Iterable[Any]:
        for segment in self._segments:
            yield from segment


def is_null_key(value: Any) -> bool:
    """空值键判定：None 视为 NULL；float('nan') 不等于自身也视为 NULL。"""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return False


def _deep_size(obj: Any, memo: set[int]) -> int:
    """递归估算对象占用字节数（容器按 id 去重，避免重复计数）。"""
    oid = id(obj)
    if oid in memo:
        return 0
    memo.add(oid)
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += _deep_size(k, memo) + _deep_size(v, memo)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            total += _deep_size(item, memo)
    return total


@dataclass
class SideData:
    """一侧输入表：列式段 + 键 -> 行号索引 + 指纹。"""

    name: str
    key: str
    columns: tuple[str, ...]
    _col_data: dict[str, SegmentedColumn] = field(default_factory=dict)
    _key_index: dict[Any, list[int]] = field(default_factory=dict)
    _segment_row_counts: list[int] = field(default_factory=list)
    null_key_rows: int = 0
    row_count: int = 0
    _fingerprint: hashlib._Hash | None = None

    def __post_init__(self) -> None:
        if self.key not in self.columns:
            raise ValueError(f"{self.name}: 键列 {self.key!r} 不在列清单 {self.columns} 中")
        self.columns = tuple(self.columns)
        self._col_data = {c: SegmentedColumn() for c in self.columns}
        self._fingerprint = hashlib.sha256()
        self._fingerprint.update(f"side|{self.name}|{self.key}|{'|'.join(self.columns)}\n".encode())

    # -- 写入 -----------------------------------------------------------
    def append_segment(self, rows: Sequence[dict[str, Any]]) -> int:
        """追加一个不可变段，返回追加后的总行数。"""
        start = self.row_count
        for col in self.columns:
            self._col_data[col].append_segment([row.get(col) for row in rows])

        key_col = self._col_data[self.key]
        for offset in range(len(rows)):
            flat_id = start + offset
            key_value = key_col[flat_id]
            if is_null_key(key_value):
                self.null_key_rows += 1
                continue
            self._key_index.setdefault(key_value, []).append(flat_id)

        for row in rows:
            self._fingerprint.update(
                repr(tuple(row.get(c) for c in self.columns)).encode("utf-8", "replace")
            )
            self._fingerprint.update(b"\n")

        self._segment_row_counts.append(len(rows))
        self.row_count += len(rows)
        return self.row_count

    # -- 读取 -----------------------------------------------------------
    def get_row(self, flat_index: int) -> dict[str, Any]:
        return {c: self._col_data[c][flat_index] for c in self.columns}

    def key_rows(self, key_value: Any) -> list[int]:
        return self._key_index.get(key_value, ())  # type: ignore[return-value]

    def key_count(self, key_value: Any) -> int:
        return len(self._key_index.get(key_value, ()))

    def keys(self) -> Iterable[Any]:
        return self._key_index.keys()

    def distinct_key_count(self) -> int:
        return len(self._key_index)

    def fingerprint_hex(self) -> str:
        return self._fingerprint.hexdigest()  # type: ignore[union-attr]

    def memory_bytes(self) -> int:
        memo: set[int] = set()
        total = _deep_size(self._col_data, memo)
        total += _deep_size(self._key_index, memo)
        total += _deep_size(self._segment_row_counts, memo)
        return total

    def segment_count(self) -> int:
        return len(self._segment_row_counts)
