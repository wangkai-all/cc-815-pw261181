"""两级关联抽样器。

抽样模型（内关联，仅两侧都出现的非空键）：
1. 键级：每个键 k 的权重 w_k 与其 *可接受关联对数* 成正比；
2. 键内级：在键 k 的可接受行对中均匀抽一个行对 (lid, rid)。

于是无条件时每个关联对的入选概率严格相等，均为 1 / Σ_k nL(k)·nR(k)。

成对条件（作用于整行对）下：
* 小键（pair_count <= max_pairs_per_key）：全枚举精确接受率
  a_k = 接受对数 / 总对数，权重校正为 pair_count · a_k；
* 大热点键：不枚举全部对，改为均匀探针估计 â_k，权重校正为
  pair_count · â_k（标记为估计值，网页与记录中如实展示）；
* 探针/枚举发现接受率为 0 的键权重为 0，永不被抽中
  （条件可使整键无结果）。

模式：
* WR（with replacement，有放回）：每次抽取独立，重复源行凭借独立
  行号身份可以反复出现；
* WOR（without replacement，无放回）：Fenwick 树按存活对扣权，
  键内排除已抽行对；精确权重下任意存活对的条件入选概率恒等。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from .fenwick import IntFenwick, FloatFenwick
from .snapshot import JoinSnapshot
from .store import _deep_size

SamplingMode = Literal["WR", "WOR"]


class ResourceLimitExceeded(RuntimeError):
    """构建期触及调用方设定的资源上限（求值次数 / 时间 / 内存）。"""

    def __init__(self, limit: str, budget: Any, used: Any, partial: dict[str, Any]):
        super().__init__(f"资源上限被突破: {limit}（预算 {budget}，已用 {used}）")
        self.limit = limit
        self.budget = budget
        self.used = used
        self.partial = partial


class SamplingError(RuntimeError):
    """抽样过程本身的非法状态（如 WOR 请求量超过可接受对总数）。"""


class _KeyExhausted(Exception):
    """内部信号：某键在 WOR 下键内已无存活接受对，调用方应清零其权重并重抽。"""

    def __init__(self, key: Any) -> None:
        super().__init__(f"键 {key!r} 的存活接受对已耗尽")
        self.key = key


@dataclass
class SamplerConfig:
    sample_size: int = 100
    mode: SamplingMode = "WR"
    # 单键关联对不超过该值时全枚举求精确接受率；否则走探针估计。
    max_pairs_per_key: int = 100_000
    # 估计键的探针数上下限。
    estimator_probes: int = 2000
    estimator_min_probes: int = 50
    # 构建期资源上限（None 表示不限制）。
    max_build_evaluations: int | None = None
    max_build_seconds: float | None = None
    max_memory_bytes: int | None = None
    # 允许物化的「已接受行对 id」缓存总量（精确列表 + 估计键蓄水池）。
    max_accepted_cache_pairs: int | None = 200_000
    rng_seed: int | None = None
    # 键内拒绝抽样的安全尝试上限（按接受率自适应放大）。
    rejection_attempt_cap: int = 10_000

    def validate(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size 必须为正整数")
        if self.mode not in ("WR", "WOR"):
            raise ValueError("mode 只能是 'WR' 或 'WOR'")
        if self.max_pairs_per_key <= 0 or self.estimator_probes <= 0:
            raise ValueError("枚举上限与探针数必须为正")
        if self.estimator_min_probes <= 0:
            raise ValueError("estimator_min_probes 必须为正")
        if self.rejection_attempt_cap < 1:
            raise ValueError("rejection_attempt_cap 必须 >= 1")
        if (
            self.max_accepted_cache_pairs is not None
            and self.max_accepted_cache_pairs <= 0
        ):
            raise ValueError("max_accepted_cache_pairs 为空或正整数")
        for name, value in (
            ("max_build_evaluations", self.max_build_evaluations),
            ("max_build_seconds", self.max_build_seconds),
            ("max_memory_bytes", self.max_memory_bytes),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} 为空或正数")


@dataclass
class KeyBuildInfo:
    """单键构建结果（检查点序列化的最小单元）。"""

    key: Any
    left_rows: int
    right_rows: int
    pair_count: int
    weight_mode: Literal["unconditioned", "exact", "estimate"]
    acceptance_rate: float          # 精确或估计的接受率（可能经 add-one 平滑）
    rate_is_estimate: bool
    rate_smoothed: bool             # 探针零命中后做了保守平滑（P1-4）
    probes_used: int                # 枚举模式 = pair_count，无条件 = 0
    accepted_count: int             # 精确接受对数；估计模式为估计接受对数(四舍五入)
    accepted_observed: int          # 探针中实际观察到的接受数
    weight: float                   # 校正后权重（浮点，建 Fenwick 时按需取整）
    accepted_pairs: list[list[int]] | None  # 完整列表或构建期蓄水池
    cache_is_reservoir: bool
    draws: int = 0                  # 该键已抽次数（恢复时用）
    # 运行期惰性构建的 WOR 存活接受对缓存（不写入检查点，恢复后按需重建）。
    alive_cache: list[list[int]] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        # alive_cache 是运行期可重建的派生物，不进入检查点。
        data.pop("alive_cache", None)
        return data

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "KeyBuildInfo":
        data = dict(data)
        data.pop("alive_cache", None)
        # 兼容旧检查点缺少新增字段的情况。
        data.setdefault("rate_smoothed", False)
        return cls(**data)


class JoinSampler:
    """从同一 JoinSnapshot 构建并抽样；支持检查点恢复后继续追加批次。"""

    def __init__(self, snapshot: JoinSnapshot, config: SamplerConfig):
        config.validate()
        self.snapshot = snapshot
        self.config = config
        self.rng = random.Random(config.rng_seed)
        self._built = False
        self._infos: list[KeyBuildInfo] = []
        self._index: dict[Any, int] = {}
        self._fenwick: IntFenwick | FloatFenwick | None = None
        self._float_weights = False
        self._initial_total_weight = 0.0
        self._drawn_pairs: set[tuple[int, int]] = set()
        self._records: list[dict[str, Any]] = []
        self._build_stats: dict[str, Any] = {}
        self._start_time = 0.0
        self._evaluations = 0
        self._cache_used = 0

    # ==================================================================
    # 构建：逐键统计接受率与校正权重
    # ==================================================================
    def build(self) -> "JoinSampler":
        if self._built:
            return self
        cfg = self.config
        self._start_time = time.monotonic()
        join_keys = self.snapshot.keys()
        per_key_cache = self._plan_cache_budget(len(join_keys))

        for k in join_keys:
            stat = self.snapshot.stat(k)
            if self.snapshot.condition.is_none:
                info = KeyBuildInfo(
                    key=k, left_rows=stat.left_rows, right_rows=stat.right_rows,
                    pair_count=stat.pair_count, weight_mode="unconditioned",
                    acceptance_rate=1.0, rate_is_estimate=False, rate_smoothed=False,
                    probes_used=0,
                    accepted_count=stat.pair_count, accepted_observed=stat.pair_count,
                    weight=float(stat.pair_count), accepted_pairs=None,
                    cache_is_reservoir=False,
                )
            elif stat.pair_count <= cfg.max_pairs_per_key:
                info = self._build_exact(k, stat.pair_count, per_key_cache)
            else:
                info = self._build_estimator(k, stat.pair_count, per_key_cache)
            self._infos.append(info)
            self._index[k] = len(self._infos) - 1
            self._check_resource_limits()

        self._float_weights = any(i.rate_is_estimate for i in self._infos)
        self._fenwick = self._make_fenwick([i.weight for i in self._infos])
        self._initial_total_weight = self._fenwick.total
        self._built = True
        self._build_stats = self._collect_build_stats()
        return self

    def _plan_cache_budget(self, key_num: int) -> int:
        cap = self.config.max_accepted_cache_pairs
        if cap is None or key_num == 0:
            return 10**12
        return max(16, cap // max(1, key_num))

    def _build_exact(self, key: Any, pair_count: int, cache_budget: int) -> KeyBuildInfo:
        left_ids = self.snapshot.left_row_ids(key)
        right_ids = self.snapshot.right_row_ids(key)
        nr = len(right_ids)
        accepted = 0
        pairs: list[list[int]] = []
        full_cache = pair_count <= cache_budget
        reservoir: list[list[int]] = []
        seen = 0
        for pid in range(pair_count):
            lid = left_ids[pid // nr]
            rid = right_ids[pid % nr]
            self._evaluations += 1
            if self._evaluate(lid, rid):
                accepted += 1
                seen += 1
                if full_cache:
                    pairs.append([lid, rid])
                elif len(reservoir) < cache_budget:
                    reservoir.append([lid, rid])
                else:
                    j = self.rng.randrange(seen)
                    if j < cache_budget:
                        reservoir[j] = [lid, rid]
        rate = accepted / pair_count if pair_count else 0.0
        self._cache_used += len(pairs) if full_cache else len(reservoir)
        return KeyBuildInfo(
            key=key, left_rows=len(left_ids), right_rows=nr, pair_count=pair_count,
            weight_mode="exact", acceptance_rate=rate, rate_is_estimate=False,
            rate_smoothed=False,
            probes_used=pair_count, accepted_count=accepted, accepted_observed=accepted,
            weight=pair_count * rate,
            accepted_pairs=pairs if full_cache else (reservoir or None),
            cache_is_reservoir=not full_cache,
        )

    def _build_estimator(self, key: Any, pair_count: int, cache_budget: int) -> KeyBuildInfo:
        cfg = self.config
        probes = min(pair_count, max(cfg.estimator_min_probes, cfg.estimator_probes))
        left_ids = self.snapshot.left_row_ids(key)
        right_ids = self.snapshot.right_row_ids(key)
        nr = len(right_ids)
        accepted = 0
        distinct_accepted = 0
        seen_pairs: set[tuple[int, int]] = set()
        reservoir: list[list[int]] = []
        for _ in range(probes):
            pid = self.rng.randrange(pair_count)
            lid = left_ids[pid // nr]
            rid = right_ids[pid % nr]
            self._evaluations += 1
            if self._evaluate(lid, rid):
                accepted += 1
                pair = (lid, rid)
                if pair in seen_pairs:
                    continue
                # 蓄水池只在“不同的接受对”流上维护，避免重复探针占据槽位（P2-3）。
                seen_pairs.add(pair)
                distinct_accepted += 1
                if len(reservoir) < cache_budget:
                    reservoir.append([lid, rid])
                else:
                    j = self.rng.randrange(distinct_accepted)
                    if j < cache_budget:
                        reservoir[j] = [lid, rid]

        smoothed = accepted == 0
        if smoothed:
            # 探针零命中不代表真实接受率为 0：用 add-one 平滑给出保守的正
            # 下界权重，保留稀有接受对被抽中的机会，并显式标记（P1-4）。
            rate = 1.0 / (probes + pair_count)
            est_accepted = max(1, int(round(pair_count * rate)))
        else:
            rate = accepted / probes
            est_accepted = max(1, int(round(pair_count * rate)))
        self._cache_used += len(reservoir)
        return KeyBuildInfo(
            key=key, left_rows=len(left_ids), right_rows=nr, pair_count=pair_count,
            weight_mode="estimate", acceptance_rate=rate, rate_is_estimate=True,
            rate_smoothed=smoothed,
            probes_used=probes, accepted_count=est_accepted, accepted_observed=accepted,
            weight=pair_count * rate,
            accepted_pairs=reservoir or None, cache_is_reservoir=True,
        )

    def _evaluate(self, left_row_id: int, right_row_id: int) -> bool:
        left_row = self.snapshot.get_left_row(left_row_id)
        right_row = self.snapshot.get_right_row(right_row_id)
        return self.snapshot.condition.evaluate(left_row, right_row)

    def _check_resource_limits(self) -> None:
        cfg = self.config
        if cfg.max_build_evaluations and self._evaluations > cfg.max_build_evaluations:
            raise ResourceLimitExceeded(
                "max_build_evaluations", cfg.max_build_evaluations,
                self._evaluations, self._partial_stats(),
            )
        if cfg.max_build_seconds:
            elapsed = time.monotonic() - self._start_time
            if elapsed > cfg.max_build_seconds:
                raise ResourceLimitExceeded(
                    "max_build_seconds", cfg.max_build_seconds,
                    round(elapsed, 3), self._partial_stats(),
                )
        if cfg.max_memory_bytes:
            used = self.memory_report()["sampler_bytes"]
            if used > cfg.max_memory_bytes:
                raise ResourceLimitExceeded(
                    "max_memory_bytes", cfg.max_memory_bytes, used,
                    self._partial_stats(),
                )

    def _partial_stats(self) -> dict[str, Any]:
        return {
            "keys_processed": len(self._infos),
            "evaluations": self._evaluations,
            "elapsed_seconds": round(time.monotonic() - self._start_time, 4),
            "accepted_pairs_cached": self._cache_used,
        }

    def _make_fenwick(self, weights: list[float]) -> IntFenwick | FloatFenwick:
        if not self._float_weights:
            int_weights = [int(round(w)) for w in weights]
            return IntFenwick(int_weights)
        return FloatFenwick([float(w) for w in weights])

    def _collect_build_stats(self) -> dict[str, Any]:
        exact_keys = [i for i in self._infos if i.weight_mode == "exact"]
        estimate_keys = [i for i in self._infos if i.weight_mode == "estimate"]
        zero_keys = [i for i in self._infos if i.acceptance_rate == 0.0]
        total_weight = sum(i.weight for i in self._infos)
        return {
            "join_key_count": len(self._infos),
            "exact_key_count": len(exact_keys),
            "estimate_key_count": len(estimate_keys),
            "zero_result_key_count": len(zero_keys),
            "zero_result_keys": [i.key for i in zero_keys],
            "condition_evaluations": self._evaluations,
            "build_seconds": round(time.monotonic() - self._start_time, 4),
            "total_weight": total_weight,
            "accepted_pairs_cached": self._cache_used,
            "materialized_join_rows": 0,  # 全量关联结果从不物化，恒为 0
        }

    # ==================================================================
    # 抽样
    # ==================================================================
    @property
    def total_weight(self) -> float:
        assert self._fenwick is not None
        return self._fenwick.total

    def draw_more(self, n: int) -> list[dict[str, Any]]:
        if not self._built:
            raise SamplingError("抽样器尚未 build()")
        if n <= 0:
            return []
        if self.config.mode == "WOR":
            # 精确模式下剩余可接受对数已知，提前给出清晰的容量错误；
            # 估计模式容量为估计值，由 Fenwick 抽键在权重耗尽时抛出。
            if not self._float_weights:
                already = sum(i.draws for i in self._infos)
                capacity = int(self._fenwick.total) + already
                if already + n > capacity:
                    raise SamplingError(
                        f"WOR 容量不足：还需 {n} 个不重复对，仅剩 {capacity - already} 个可接受对"
                    )
        new_records: list[dict[str, Any]] = []
        for _ in range(n):
            new_records.append(self._draw_one())
        self._records.extend(new_records)
        return new_records

    def sample(self) -> SamplingResult:
        self.build()
        if not self._records:
            self.draw_more(self.config.sample_size)
        return self.result()

    def _total_is_empty(self, total_w: float) -> bool:
        if self._float_weights:
            # 与 FloatFenwick 动态 epsilon 同一尺度，避免阈值不一致（P1-3）。
            eps = max(
                1e-12,
                1e-12 * max(1.0, abs(self._initial_total_weight)),
                1e-12 * max(1.0, abs(total_w)),
            )
            return total_w <= eps
        return total_w <= 0

    def _draw_one(self) -> dict[str, Any]:
        assert self._fenwick is not None
        if self._total_is_empty(self._fenwick.total):
            zero_keys = self._build_stats.get("zero_result_keys", [])
            raise SamplingError(
                "没有可抽样的接受对：成对条件使所有关联键的接受率均为 0"
                f"（零结果键 {len(zero_keys)} 个）"
            )

        # WOR 下被抽中的键可能在键内已无存活对（估计容量与真实接受对数偏离），
        # 此时清零该键残余权重并重抽其他键；有界重试保证终止。
        for _attempt in range(len(self._infos) + 2):
            total_w = self._fenwick.total
            if self._total_is_empty(total_w):
                raise SamplingError("所有存活接受对已抽完（WOR）")
            try:
                idx = self._fenwick.pick(self.rng)
            except IndexError as exc:  # 收敛为统一业务异常（P1-3）
                raise SamplingError("所有存活接受对已抽完（WOR）") from exc
            info = self._infos[idx]
            live_weight = self._fenwick.weight(idx)
            try:
                lid, rid, attempts, live_count = self._draw_pair_within(info)
            except _KeyExhausted:
                self._fenwick.add(idx, -live_weight)  # type: ignore[union-attr]
                continue

            # 真实条件入选概率（P2-1）：键概率取存活权重，键内取存活对数。
            p_key = (live_weight / total_w) if total_w else 0.0
            p_pair = 1.0 / live_count if live_count else 0.0

            if self.config.mode == "WOR":
                if self._float_weights:
                    step = info.weight / max(1, info.accepted_count)
                    current = self._fenwick.weight(idx)  # type: ignore[union-attr]
                    # 不扣到 0，保留相对极小的正下限：估计容量偏低时该键仍
                    # 可被抽，直到真实存活对耗尽（_KeyExhausted 时清零）。
                    floor = max(1e-12, self._initial_total_weight * 1e-9)
                    target = max(floor, current - step)
                    self._fenwick.add(idx, target - current)  # type: ignore[union-attr]
                else:
                    self._fenwick.add(idx, -1)  # type: ignore[union-attr]
                info.draws += 1
                self._drawn_pairs.add((lid, rid))
                if info.alive_cache is not None:
                    info.alive_cache = [
                        p for p in info.alive_cache
                        if not (p[0] == lid and p[1] == rid)
                    ]
            record = {
                "draw_index": len(self._records),
                "key": info.key,
                "left_row_id": lid,
                "right_row_id": rid,
                "left_row": self.snapshot.get_left_row(lid),
                "right_row": self.snapshot.get_right_row(rid),
                "mode": self.config.mode,
                "weight_mode": info.weight_mode,
                "rate_is_estimate": info.rate_is_estimate,
                "rate_smoothed": info.rate_smoothed,
                "pair_count_under_key": info.pair_count,
                "acceptance_rate": info.acceptance_rate,
                "accepted_pair_count_under_key": info.accepted_count,
                "key_weight": info.weight,
                "total_weight_at_draw": total_w,
                "p_select_key": p_key,
                "p_select_pair_within_key": p_pair,
                "p_joint": p_key * p_pair,
                "rejection_attempts": attempts,
                "condition_expression": self.snapshot.condition.expression,
                "condition_passed": True,
            }
            return record
        raise SamplingError("所有存活接受对已抽完（WOR）")

    def _draw_pair_within(self, info: KeyBuildInfo) -> tuple[int, int, int, int]:
        """键内均匀抽一个可接受行对。

        返回 (left_row_id, right_row_id, 尝试次数, 键内存活对数)。
        均匀性原则：只在“均匀探针 + 接受/未用判定”的拒绝抽样，或在完整
        存活接受对列表上的均匀选择之间取路径；绝不在有偏的小蓄水池上
        直接取对（修复 P1-1）。WOR 下键内真实耗尽抛 _KeyExhausted。
        """
        left_ids = self.snapshot.left_row_ids(info.key)
        right_ids = self.snapshot.right_row_ids(info.key)

        # 1) 无条件。
        if info.weight_mode == "unconditioned":
            if self.config.mode == "WOR":
                return self._uniform_unused_pair(info, left_ids, right_ids)
            lid = left_ids[self.rng.randrange(len(left_ids))]
            rid = right_ids[self.rng.randrange(len(right_ids))]
            return lid, rid, 1, info.pair_count

        # 2) 完整精确接受列表（非蓄水池）：在全部接受对上均匀取。
        if info.accepted_pairs is not None and not info.cache_is_reservoir:
            if self.config.mode == "WOR":
                pair, live = self._pick_unused_from_full_list(
                    info, info.accepted_pairs, left_ids, right_ids
                )
                return pair[0], pair[1], 1, live
            pair = info.accepted_pairs[self.rng.randrange(len(info.accepted_pairs))]
            return pair[0], pair[1], 1, len(info.accepted_pairs)

        # 3) 蓄水池/无缓存：纯拒绝抽样。WR 用自适应高 cap；WOR 不设会导致
        #    有偏回退的硬 cap，持续重试直到命中未用接受对，确有存活对时必然终止。
        if self.config.mode == "WR":
            return self._rejection_wr(info, left_ids, right_ids)
        return self._rejection_wor(info, left_ids, right_ids)

    # -- 拒绝抽样 -------------------------------------------------------
    def _rejection_wr(
        self, info: KeyBuildInfo, left_ids: list[int], right_ids: list[int]
    ) -> tuple[int, int, int, int]:
        nr = len(right_ids)
        rate = max(info.acceptance_rate, 1e-12)
        cap = min(self.config.rejection_attempt_cap, max(64, int(math.ceil(8.0 / rate))))
        for attempt in range(1, cap + 1):
            pid = self.rng.randrange(info.pair_count)
            lid, rid = left_ids[pid // nr], right_ids[pid % nr]
            if self._evaluate(lid, rid):
                return lid, rid, attempt, max(1, info.accepted_count)
        # 仅当接受率极低、自适应 cap 内未命中时到达。不使用有偏蓄水池，
        # 而是惰性枚举出该键全部接受对后均匀取，保持严格均匀（P1-1）。
        accepted = self._enumerate_accepted(info, left_ids, right_ids)
        if not accepted:
            raise SamplingError(
                f"键 {info.key!r} 无接受对，拒绝抽样 {cap} 次未命中"
            )
        pair = accepted[self.rng.randrange(len(accepted))]
        return pair[0], pair[1], cap, len(accepted)

    def _rejection_wor(
        self, info: KeyBuildInfo, left_ids: list[int], right_ids: list[int]
    ) -> tuple[int, int, int, int]:
        nr = len(right_ids)
        # 已有惰性存活缓存：直接在存活接受对上均匀取（严格均匀且 O(1)）。
        if info.alive_cache is not None:
            if not info.alive_cache:
                raise _KeyExhausted(info.key)
            j = self.rng.randrange(len(info.alive_cache))
            pair = info.alive_cache[j]
            return pair[0], pair[1], 1, len(info.alive_cache)

        # 自适应软上限：达到后惰性枚举一次，之后该键走 O(1) 缓存路径。
        rate = max(info.acceptance_rate, 1e-12)
        soft_cap = min(self.config.rejection_attempt_cap,
                       max(64, int(math.ceil(8.0 / rate))))
        for attempt in range(1, soft_cap + 1):
            pid = self.rng.randrange(info.pair_count)
            lid, rid = left_ids[pid // nr], right_ids[pid % nr]
            if (lid, rid) in self._drawn_pairs:
                continue
            if self._evaluate(lid, rid):
                return lid, rid, attempt, max(1, info.accepted_count)

        # 软上限到达：惰性构建“当前未使用”的存活接受对列表（只构建一次）。
        alive = self._build_alive_cache(info, left_ids, right_ids)
        if not alive:
            raise _KeyExhausted(info.key)
        pair = alive[self.rng.randrange(len(alive))]
        return pair[0], pair[1], soft_cap, len(alive)

    def _enumerate_accepted(
        self, info: KeyBuildInfo, left_ids: list[int], right_ids: list[int]
    ) -> list[list[int]]:
        """枚举该键全部接受对（仅 WR 低接受率回退时使用，结果不持久化）。"""
        nr = len(right_ids)
        out: list[list[int]] = []
        for pid in range(info.pair_count):
            lid, rid = left_ids[pid // nr], right_ids[pid % nr]
            if self._evaluate(lid, rid):
                out.append([lid, rid])
        return out

    def _build_alive_cache(
        self, info: KeyBuildInfo, left_ids: list[int], right_ids: list[int]
    ) -> list[list[int]]:
        """枚举当前未使用的接受对并缓存为存活列表，之后在其上均匀抽取。"""
        nr = len(right_ids)
        alive: list[list[int]] = []
        for pid in range(info.pair_count):
            lid, rid = left_ids[pid // nr], right_ids[pid % nr]
            if (lid, rid) in self._drawn_pairs:
                continue
            if self._evaluate(lid, rid):
                alive.append([lid, rid])
        info.alive_cache = alive
        return alive

    def _uniform_unused_pair(
        self,
        info: KeyBuildInfo,
        left_ids: list[int],
        right_ids: list[int],
    ) -> tuple[int, int, int, int]:
        """WOR 无条件：在键内全部行对中均匀取一个未使用对。"""
        nl, nr = len(left_ids), len(right_ids)
        total = nl * nr
        start = self.rng.randrange(total)
        for offset in range(total):
            pid = (start + offset) % total
            lid, rid = left_ids[pid // nr], right_ids[pid % nr]
            if (lid, rid) not in self._drawn_pairs:
                return lid, rid, 1, total - info.draws
        raise _KeyExhausted(info.key)

    def _pick_unused_from_full_list(
        self,
        info: KeyBuildInfo,
        pairs: list[list[int]],
        left_ids: list[int],
        right_ids: list[int],
    ) -> tuple[list[int], int]:
        """WOR 完整精确列表：均匀取未用对；若列表项已全部用过则惰性重建。"""
        order = self.rng.sample(range(len(pairs)), len(pairs))
        for j in order:
            pair = pairs[j]
            if (pair[0], pair[1]) not in self._drawn_pairs:
                live = sum(
                    1 for p in pairs if (p[0], p[1]) not in self._drawn_pairs
                )
                return pair, live
        # 完整列表理论上覆盖全部接受对，走到这里说明已抽尽。
        raise _KeyExhausted(info.key)

    # ==================================================================
    # 结果、内存与恢复状态
    # ==================================================================
    def result(self) -> "SamplingResult":
        if not self._built:
            raise SamplingError("抽样器尚未 build()")
        key_counts: dict[Any, int] = {}
        for rec in self._records:
            key_counts[rec["key"]] = key_counts.get(rec["key"], 0) + 1
        return SamplingResult(
            records=list(self._records),
            key_selection_counts=key_counts,
            build_stats=dict(self._build_stats),
            memory_report=self.memory_report(),
            config=asdict(self.config),
            snapshot_fingerprint=self.snapshot.fingerprint,
            snapshot_describe=self.snapshot.describe(),
            rng_seed=self.config.rng_seed,
        )

    def memory_report(self) -> dict[str, Any]:
        memo: set[int] = set()
        sampler_bytes = _deep_size(self._infos, memo)
        if self._fenwick is not None:
            sampler_bytes += _deep_size(self._fenwick, memo)
        sampler_bytes += _deep_size(self._drawn_pairs, memo)
        return {
            "store_bytes": self.snapshot.left.memory_bytes()
            + self.snapshot.right.memory_bytes(),
            "sampler_bytes": sampler_bytes,
            "accepted_pairs_cached": self._cache_used,
            "materialized_join_rows": 0,
        }

    # -- 检查点状态 -----------------------------------------------------
    def export_state(self) -> dict[str, Any]:
        if not self._built:
            raise SamplingError("仅已构建的抽样器可导出状态")
        remaining_weights: list[float] = []
        for i, info in enumerate(self._infos):
            if self._float_weights:
                remaining_weights.append(float(self._fenwick.weight(i)))  # type: ignore[union-attr]
            else:
                remaining_weights.append(float(self._fenwick.weight(i)))  # type: ignore[union-attr]
        return {
            "version": 1,
            "config": asdict(self.config),
            "snapshot_fingerprint": self.snapshot.fingerprint,
            "snapshot_describe": self.snapshot.describe(),
            "condition_expression": self.snapshot.condition.expression,
            "rng_seed": self.config.rng_seed,
            "rng_state": _rng_state_to_jsonable(self.rng.getstate()),
            "float_weights": self._float_weights,
            "initial_total_weight": float(self._initial_total_weight),
            "key_order": [i.key for i in self._infos],
            "key_build": [i.to_jsonable() for i in self._infos],
            "remaining_weights": remaining_weights,
            "drawn_pairs": [list(p) for p in sorted(self._drawn_pairs)],
            "records": self._records,
            "build_stats": self._build_stats,
            "build_evaluations": self._evaluations,
            "cache_used": self._cache_used,
        }

    @classmethod
    def restore(cls, state: dict[str, Any], snapshot: JoinSnapshot) -> "JoinSampler":
        if state["snapshot_fingerprint"] != snapshot.fingerprint:
            from .checkpoint import ResumeError

            raise ResumeError("快照指纹不一致：当前数据与检查点不是同一份快照，拒绝恢复")
        if state.get("rng_seed") != state["config"].get("rng_seed"):
            from .checkpoint import ResumeError

            raise ResumeError("检查点内部种子字段不一致，文件可能已损坏")
        # 配置字段健全性（P3-2）：检查点必须携带一份完整、可构造的配置；
        # 字段缺失/多出或取值非法都拒绝，而不是静默沿用损坏值。
        # 与“当前期望配置”的逐项等值校验见 restore_sampler(..., expect_config)。
        config_data = state["config"]
        expected_fields = set(SamplerConfig.__dataclass_fields__)
        actual_fields = set(config_data)
        missing = expected_fields - actual_fields
        unknown = actual_fields - expected_fields
        if missing or unknown:
            from .checkpoint import ResumeError

            raise ResumeError(
                f"检查点配置字段不兼容（缺失 {sorted(missing)}，多出 {sorted(unknown)}）"
            )
        cfg = SamplerConfig(**config_data)
        cfg.validate()
        sampler = cls(snapshot, cfg)
        sampler._built = True
        sampler._float_weights = state["float_weights"]
        # 旧检查点缺少该字段时，用剩余权重之和作保守回退（尺度只会偏小）。
        sampler._initial_total_weight = float(
            state.get("initial_total_weight", sum(state["remaining_weights"]))
        )
        sampler._infos = [KeyBuildInfo.from_jsonable(d) for d in state["key_build"]]
        sampler._index = {info.key: i for i, info in enumerate(sampler._infos)}
        if [i.key for i in sampler._infos] != state["key_order"]:
            from .checkpoint import ResumeError

            raise ResumeError("键顺序与检查点记录不一致")
        if state["float_weights"]:
            sampler._fenwick = FloatFenwick(state["remaining_weights"])
        else:
            sampler._fenwick = IntFenwick([int(round(w)) for w in state["remaining_weights"]])
        sampler.rng.setstate(_rng_state_from_jsonable(state["rng_state"]))
        sampler._drawn_pairs = {tuple(p) for p in state["drawn_pairs"]}
        sampler._records = list(state["records"])
        sampler._build_stats = dict(state["build_stats"])
        sampler._evaluations = state["build_evaluations"]
        sampler._cache_used = state["cache_used"]
        for info in sampler._infos:
            pass  # draws 已在 KeyBuildInfo 中随检查点保存
        return sampler


@dataclass
class SamplingResult:
    records: list[dict[str, Any]]
    key_selection_counts: dict[Any, int]
    build_stats: dict[str, Any]
    memory_report: dict[str, Any]
    config: dict[str, Any]
    snapshot_fingerprint: str
    snapshot_describe: dict[str, Any]
    rng_seed: int | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "key_selection_counts": [
                {"key": k, "count": c} for k, c in sorted(
                    self.key_selection_counts.items(), key=lambda kv: -kv[1]
                )
            ],
            "build_stats": self.build_stats,
            "memory_report": self.memory_report,
            "config": self.config,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "snapshot_describe": self.snapshot_describe,
            "rng_seed": self.rng_seed,
            "sample_size": len(self.records),
        }


def _rng_state_to_jsonable(state: tuple) -> list:
    # random.Random.getstate() -> (version, internaltuple, gauss_next)
    version, internal, gauss_next = state
    return [version, list(internal), gauss_next]


def _rng_state_from_jsonable(data: list) -> tuple:
    version, internal, gauss_next = data
    return (version, tuple(internal), gauss_next)
