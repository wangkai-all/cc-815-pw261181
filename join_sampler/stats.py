"""统计辅助与版本对比。

仅依赖标准库实现卡方检验所需的正则化下/上不完全 Gamma 函数，
避免引入 scipy 等重依赖。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

_SQRT_EPS = math.sqrt(2.220446049250313e-16)


def _gammap_series(a: float, x: float) -> float:
    """正则化下不完全 Gamma P(a, x)，级数表示（x < a+1 时稳定）。"""
    if x <= 0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(2000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _SQRT_EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaq_cf(a: float, x: float) -> float:
    """正则化上不完全 Gamma Q(a, x)，连分式表示。"""
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 2001):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _SQRT_EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(stat: float, df: int) -> float:
    """卡方分布上尾概率 P(X² >= stat)，即拟合优度检验的 p 值。"""
    if stat <= 0 or df <= 0:
        return 1.0
    a = df / 2.0
    x = stat / 2.0
    if x < a + 1.0:
        return 1.0 - _gammap_series(a, x)
    return _gammaq_cf(a, x)


def chi_square_uniform(
    observed: Sequence[int], expected_prob: Sequence[float]
) -> tuple[float, float, int]:
    """对分类观测做卡方检验，返回 (统计量, p 值, 合并后组数)。

    期望数 < 5 的相邻组会被合并，保证近似有效。
    """
    if len(observed) == 0:
        raise ValueError("observed 不能为空")
    if len(observed) != len(expected_prob):
        raise ValueError(
            f"observed 与 expected_prob 长度不一致: {len(observed)} != {len(expected_prob)}"
        )
    if any(o < 0 for o in observed):
        raise ValueError("observed 不能包含负计数")
    if any(p < 0.0 for p in expected_prob):
        raise ValueError("expected_prob 不能包含负概率")
    prob_sum = math.fsum(expected_prob)
    if not math.isfinite(prob_sum) or abs(prob_sum - 1.0) > 1e-9:
        raise ValueError(f"expected_prob 之和必须为 1，实际为 {prob_sum!r}")
    total = sum(observed)
    merged_obs: list[int] = []
    merged_exp: list[float] = []
    acc_o, acc_e = 0, 0.0
    for o, p in zip(observed, expected_prob):
        acc_o += o
        acc_e += total * p
        if acc_e >= 5.0:
            merged_obs.append(acc_o)
            merged_exp.append(acc_e)
            acc_o, acc_e = 0, 0.0
    if acc_e > 0:
        if merged_obs:
            merged_obs[-1] += acc_o
            merged_exp[-1] += acc_e
        else:
            merged_obs, merged_exp = [acc_o], [acc_e]
    k = len(merged_obs)
    if k < 2:
        return 0.0, 1.0, k
    stat = sum((o - e) ** 2 / e for o, e in zip(merged_obs, merged_exp))
    return stat, chi2_sf(stat, k - 1), k


def _binomial_logpmf(k: int, n: int, p: float) -> float:
    """log P(X=k)，用 lgamma 避免大组合数转浮点时溢出。"""
    if p == 0.0:
        return 0.0 if k == 0 else float("-inf")
    if p == 1.0:
        return 0.0 if k == n else float("-inf")
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binomial_two_sided_p(k: int, n: int, p: float) -> float:
    """二项分布双侧 p 值（精确枚举概率不高于观测概率的取值，log 域求和）。"""
    if n < 0:
        raise ValueError("n 必须为非负整数")
    if not isinstance(k, int) or k < 0 or k > n:
        raise ValueError(f"k 必须是 [0, n] 内的整数，收到 k={k}, n={n}")
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"p 必须在 [0, 1] 内，收到 {p!r}")
    log_pmf_obs = _binomial_logpmf(k, n, p)
    total = 0.0
    for j in range(n + 1):
        log_pmf = _binomial_logpmf(j, n, p)
        if log_pmf <= log_pmf_obs + 1e-12:
            total += math.exp(log_pmf)
    return min(1.0, total)


def compare_runs(
    base: dict[str, Any], other: dict[str, Any]
) -> dict[str, Any]:
    """比较两版抽样结果（追加批次前后或两份不同运行）。"""
    base_records = base["records"]
    other_records = other["records"]

    def pair_set(records: list[dict[str, Any]]) -> set[tuple[int, int]]:
        return {(r["left_row_id"], r["right_row_id"]) for r in records}

    set_a = pair_set(base_records)
    set_b = pair_set(other_records)
    base_counts = _count_by_key(base_records)
    other_counts = _count_by_key(other_records)
    all_keys = sorted(set(base_counts) | set(other_counts), key=repr)
    return {
        "base_sample_size": len(base_records),
        "other_sample_size": len(other_records),
        "shared_pair_count": len(set_a & set_b),
        "base_only_pair_count": len(set_a - set_b),
        "other_only_pair_count": len(set_b - set_a),
        "key_distribution": [
            {
                "key": k,
                "base_count": base_counts.get(k, 0),
                "other_count": other_counts.get(k, 0),
                "delta": other_counts.get(k, 0) - base_counts.get(k, 0),
            }
            for k in all_keys
        ],
        "same_snapshot": base.get("snapshot_fingerprint")
        == other.get("snapshot_fingerprint"),
    }


def _count_by_key(records: list[dict[str, Any]]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for r in records:
        counts[r["key"]] = counts.get(r["key"], 0) + 1
    return counts
