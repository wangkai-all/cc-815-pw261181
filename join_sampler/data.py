"""输入数据装载：CSV -> SideData，以及合成数据生成器。

CSV 规则：
* 首行为表头；类型自动推断（int / float / bool / None / 字符串），
  空字段与字面量 null/NULL/NaN 视为 NULL；
* 可一次读取，也可分批追加为不同段（段不可变，快照不随后续追加变化）；
* 合成生成器用于在网页上快速造热点键/空值键等场景。
"""

from __future__ import annotations

import csv
from typing import Any, Iterable

from .store import SideData

_NULL_TOKENS = {"", "null", "none", "nan", "na"}
_TRUE_TOKENS = {"true", "yes"}
_FALSE_TOKENS = {"false", "no"}


def infer_scalar(token: str) -> Any:
    lowered = token.strip().lower()
    if lowered in _NULL_TOKENS:
        return None
    if lowered in _TRUE_TOKENS:
        return True
    if lowered in _FALSE_TOKENS:
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def load_csv_side(
    path: str,
    *,
    name: str,
    key: str,
    batch_rows: int | None = None,
) -> SideData:
    """读取 CSV 为 SideData；batch_rows 给定时按该行数切分为多个段。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV 缺少表头")
        columns = tuple(reader.fieldnames)
        side = SideData(name=name, key=key, columns=columns)
        batch: list[dict[str, Any]] = []
        for raw in reader:
            row = {col: infer_scalar(raw[col]) for col in columns}
            batch.append(row)
            if batch_rows and len(batch) >= batch_rows:
                side.append_segment(batch)
                batch = []
        if batch:
            side.append_segment(batch)
    return side


def side_from_rows(
    rows: Iterable[dict[str, Any]], *, name: str, key: str
) -> SideData:
    """从内存中的字典行构造 SideData（单段）。"""
    rows = list(rows)
    if not rows:
        raise ValueError(f"{name}: 至少需要一行数据")
    columns = tuple(rows[0].keys())
    side = SideData(name=name, key=key, columns=columns)
    side.append_segment(rows)
    return side


def make_synthetic_tables(
    *,
    left_rows: int = 50_000,
    right_rows: int = 50_000,
    distinct_keys: int = 500,
    hot_key_multiplier: int = 200,
    null_key_fraction: float = 0.02,
    value_max: int = 10_000,
    rng_seed: int = 42,
) -> tuple[SideData, SideData]:
    """生成一对合成表。

    键分布：
    * 第 0 号键为热点键，按 hot_key_multiplier 加权出现；
    * 其余键近似均匀；
    * 按 null_key_fraction 注入 NULL 键行；
    数值列 v 用于成对条件（如 abs(L.v - R.v) <= k）。
    """
    import random

    rng = random.Random(rng_seed)
    weights = [hot_key_multiplier] + [1] * (distinct_keys - 1)

    def gen(n: int, tag: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i in range(n):
            if rng.random() < null_key_fraction:
                kv: Any = None
            else:
                kv = rng.choices(range(distinct_keys), weights=weights, k=1)[0]
            rows.append(
                {
                    "k": kv,
                    "v": rng.randrange(value_max),
                    "src": f"{tag}-{i}",
                }
            )
        return rows

    left = SideData(name="left", key="k", columns=("k", "v", "src"))
    left.append_segment(gen(left_rows, "L"))
    right = SideData(name="right", key="k", columns=("k", "v", "src"))
    right.append_segment(gen(right_rows, "R"))
    return left, right
