"""受限缓存 / 估计路径 / 安全加固的边界回归测试。

补齐主测试集未强制覆盖的场景：
* 缓存预算 < 接受对数时，精确键走蓄水池路径，WR 仍严格均匀、
  WOR 可耗尽全部接受对且边际均匀（P1-1/P1-2 回归）；
* 估计热点键小缓存 WOR 不崩溃、不重复，并惰性构建存活缓存；
* 探针零命中的 add-one 平滑保留稀有键被抽中的机会（P1-4）；
* FloatFenwick 动态 epsilon 不扼杀相对总量极小但合法的幸存键（P1-3）；
* 条件沙箱下划线属性封堵与 pow 资源上限（P2-4/P2-5）；
* stats 入参校验（P2-6）、配置校验（P3-3）、异构 tuple 键排序（P3-3）；
* 损坏检查点收敛为 ResumeError、完整配置等值校验（P3-1/P3-2）。
"""

import json
import os
import tempfile
import unittest
from collections import Counter

from join_sampler.checkpoint import (
    ResumeError,
    load_checkpoint,
    restore_sampler,
    save_checkpoint,
)
from join_sampler.conditions import Condition, ConditionError
from join_sampler.data import side_from_rows
from join_sampler.fenwick import FloatFenwick
from join_sampler.sampler import JoinSampler, SamplerConfig, SamplingError
from join_sampler.snapshot import build_snapshot
from join_sampler.stats import binomial_two_sided_p, chi_square_uniform


def _single_key_snapshot(expression, nl=20, nr=20, key=1):
    left = [{"k": key, "v": i} for i in range(nl)]
    right = [{"k": key, "v": j} for j in range(nr)]
    return build_snapshot(
        side_from_rows(left, name="l", key="k"),
        side_from_rows(right, name="r", key="k"),
        Condition.compile(expression),
    )


def _enumerate_accepted(snap):
    out = []
    for k in snap.keys():
        for lid in snap.left_row_ids(k):
            for rid in snap.right_row_ids(k):
                if snap.condition.evaluate(
                    snap.get_left_row(lid), snap.get_right_row(rid)
                ):
                    out.append((lid, rid))
    return out


class BoundedCacheExactKeyTests(unittest.TestCase):
    """精确枚举键但缓存预算不足（cache_is_reservoir=True）。"""

    def setUp(self):
        # 20x20=400 对，条件 i+j<=15 恰好接受 136 对。
        self.snap = _single_key_snapshot("L.v + R.v <= 15")
        self.accepted = _enumerate_accepted(self.snap)
        self.assertEqual(len(self.accepted), 136)

    def _small_cache_config(self, **over):
        cfg = dict(
            mode="WR", rng_seed=11, max_pairs_per_key=10_000,
            max_accepted_cache_pairs=16,  # 预算 16 << 136 接受对
        )
        cfg.update(over)
        return SamplerConfig(**cfg)

    def test_reservoir_path_actually_forced(self):
        sampler = JoinSampler(self.snap, self._small_cache_config()).build()
        info = sampler._infos[0]
        self.assertEqual(info.weight_mode, "exact")
        self.assertTrue(info.cache_is_reservoir)
        self.assertLessEqual(len(info.accepted_pairs or []), 16)

    def test_wr_bounded_cache_still_uniform(self):
        sampler = JoinSampler(self.snap, self._small_cache_config()).build()
        records = sampler.draw_more(4000)
        counts = Counter((r["left_row_id"], r["right_row_id"]) for r in records)
        observed = [counts.get(p, 0) for p in self.accepted]
        _, p_value, _ = chi_square_uniform(
            observed, [1.0 / len(self.accepted)] * len(self.accepted)
        )
        self.assertGreater(p_value, 0.01, f"受限缓存 WR 不均匀 p={p_value:.4f}")

    def test_wor_exhausts_every_accepted_pair_without_biased_fallback(self):
        sampler = JoinSampler(
            self.snap, self._small_cache_config(mode="WOR")
        ).build()
        m = len(self.accepted)
        records = sampler.draw_more(m)  # 旧实现会在此崩溃或只给蓄水池内的对
        pairs = [(r["left_row_id"], r["right_row_id"]) for r in records]
        self.assertEqual(len(set(pairs)), m)
        self.assertEqual(set(pairs), set(self.accepted))
        self.assertTrue(all(r["condition_passed"] for r in records))
        # 再抽一个必然报容量错误，而不是死循环。
        with self.assertRaises(SamplingError):
            sampler.draw_more(1)

    def test_wor_bounded_cache_marginal_uniformity_across_runs(self):
        total = Counter()
        runs, draw_n = 120, 34  # 每轮抽 1/4，独立种子，边际概率应对称相等
        for seed in range(runs):
            sampler = JoinSampler(
                self.snap,
                self._small_cache_config(mode="WOR", rng_seed=1000 + seed),
            ).build()
            for r in sampler.draw_more(draw_n):
                total[(r["left_row_id"], r["right_row_id"])] += 1
        observed = [total.get(p, 0) for p in self.accepted]
        _, p_value, _ = chi_square_uniform(
            observed, [1.0 / len(self.accepted)] * len(self.accepted)
        )
        self.assertGreater(p_value, 0.01, f"受限缓存 WOR 边际不均匀 p={p_value:.4f}")


class EstimatorBoundedCacheWorTests(unittest.TestCase):
    def setUp(self):
        # 100x100=10000 对的热点键，条件接受 688 对，强制走估计路径。
        self.snap = _single_key_snapshot("abs(L.v - R.v) <= 3", nl=100, nr=100)
        self.accepted = _enumerate_accepted(self.snap)
        self.assertEqual(len(self.accepted), 688)

    def _sampler(self, condition=None):
        snap = self.snap if condition is None else _single_key_snapshot(
            condition, nl=100, nr=100
        )
        return JoinSampler(
            snap,
            SamplerConfig(
                sample_size=300, mode="WOR", rng_seed=55,
                max_pairs_per_key=2000, estimator_probes=800,
                max_accepted_cache_pairs=16,
            ),
        ).build()

    def test_estimator_wor_small_cache_no_crash_no_duplicate(self):
        sampler = self._sampler()
        info = sampler._infos[0]
        self.assertTrue(info.rate_is_estimate)
        records = sampler.draw_more(300)  # 旧实现：蓄水池 16 抽完即抛错
        pairs = [(r["left_row_id"], r["right_row_id"]) for r in records]
        self.assertEqual(len(set(pairs)), 300)
        self.assertTrue(set(pairs).issubset(set(self.accepted)))

    def test_estimator_wor_exhaustion_builds_lazy_alive_cache(self):
        # 抽到接近全部 688 个接受对：拒绝抽样必然多次触顶，惰性枚举存活对。
        sampler = self._sampler()
        info = sampler._infos[0]
        records = sampler.draw_more(688)
        pairs = {(r["left_row_id"], r["right_row_id"]) for r in records}
        self.assertEqual(len(pairs), 688)
        self.assertEqual(pairs, set(self.accepted))
        self.assertIsNotNone(info.alive_cache)  # 触顶后构建的存活缓存
        with self.assertRaises(SamplingError):
            sampler.draw_more(1)

    def test_estimator_wor_low_acceptance_rate_triggers_cache(self):
        # 接受率约 1.98% 且把拒绝 cap 压到 4：几乎必然触顶，
        # 验证低接受率路径下存活缓存被构建且抽对仍全部合法。
        sampler = JoinSampler(
            _single_key_snapshot("abs(L.v - R.v) <= 0", nl=100, nr=100),
            SamplerConfig(
                sample_size=80, mode="WOR", rng_seed=55,
                max_pairs_per_key=2000, estimator_probes=800,
                max_accepted_cache_pairs=16, rejection_attempt_cap=4,
            ),
        ).build()
        accepted = _enumerate_accepted(sampler.snapshot)
        self.assertEqual(len(accepted), 100)
        records = sampler.draw_more(80)
        self.assertIsNotNone(sampler._infos[0].alive_cache)
        pairs = {(r["left_row_id"], r["right_row_id"]) for r in records}
        self.assertEqual(len(pairs), 80)
        self.assertTrue(pairs.issubset(set(accepted)))


class ZeroHitSmoothingTests(unittest.TestCase):
    def _build_smoothed_sampler(self, seed):
        left, right = [], []
        for i in range(20):  # 键 1：20x20=400 对，条件恒真
            left.append({"k": 1, "v": i})
            right.append({"k": 1, "v": i})
        for i in range(100):  # 键 2：100x100=10000 对，仅 1 对接受
            left.append({"k": 2, "v": i})
            right.append({"k": 2, "v": i})
        for i in range(10):  # 键 3：10x10=100 对，条件恒假
            left.append({"k": 3, "v": i})
            right.append({"k": 3, "v": i})
        snap = build_snapshot(
            side_from_rows(left, name="l", key="k"),
            side_from_rows(right, name="r", key="k"),
            Condition.compile("L.k == 1 or (L.k == 2 and L.v == 99 and R.v == 99)"),
        )
        sampler = JoinSampler(
            snap,
            SamplerConfig(
                sample_size=10, mode="WR", rng_seed=seed,
                max_pairs_per_key=2000, estimator_probes=50,
                estimator_min_probes=50,
            ),
        ).build()
        return sampler

    def test_smoothed_rare_key_stays_reachable(self):
        # 探针仅 50 次、命中率 1/10000，逐种子寻找零命中（平滑）情形。
        sampler = None
        for seed in range(300):
            candidate = self._build_smoothed_sampler(seed)
            info2 = candidate._infos[candidate._index[2]]
            if info2.rate_smoothed:
                sampler = candidate
                break
        self.assertIsNotNone(sampler, "300 个种子内未出现探针零命中，测试前提失效")
        info2 = sampler._infos[sampler._index[2]]
        self.assertTrue(info2.rate_is_estimate)
        self.assertEqual(info2.accepted_observed, 0)
        self.assertGreater(info2.weight, 0.0)
        info3 = sampler._infos[sampler._index[3]]
        self.assertEqual(info3.weight, 0.0)
        self.assertFalse(info3.rate_smoothed)

        records = sampler.draw_more(6000)
        rare = [r for r in records if r["key"] == 2]
        self.assertGreater(len(rare), 0, "平滑后稀有键仍应有机会被抽中")
        # 抽中的只能是那唯一真实接受对（行内容 v 都是 99）。
        for r in rare:
            self.assertEqual(r["left_row"]["v"], 99)
            self.assertEqual(r["right_row"]["v"], 99)
            self.assertTrue(r["rate_smoothed"])
        self.assertTrue(all(r["key"] != 3 for r in records))


class FloatFenwickEpsilonTests(unittest.TestCase):
    def test_survivor_weight_above_floor_not_killed(self):
        # 复刻真实热点场景：初始总量 5e9（数十亿对），WOR 把两键都扣到
        # floor = 总量*1e-9 = 5e3 量级。固定 epsilon 会用陈旧大阈值误杀，
        # 动态 epsilon 取 max(eps0, 当前尺度) 后两幸存键都必须可抽（P1-3）。
        fw = FloatFenwick([2.5e9, 2.5e9])
        fw.add(0, -(2.5e9 - 5.0e3))
        fw.add(1, -(2.5e9 - 5.0e3))
        self.assertAlmostEqual(fw.total, 1.0e4)
        rng = __import__("random").Random(0)
        hits = {fw.pick(rng) for _ in range(60)}
        self.assertEqual(hits, {0, 1})  # 两个幸存键都仍可抽
        fw.add(0, -5e3)
        fw.add(1, -5e3)
        with self.assertRaises(IndexError):
            fw.pick(rng)

    def test_small_scale_weights_pickable(self):
        fw = FloatFenwick([1e-6, 2e-6])
        idx = fw.pick(__import__("random").Random(1))
        self.assertIn(idx, (0, 1))
        fw.add(0, -1e-6)
        self.assertEqual(fw.pick(__import__("random").Random(1)), 1)


class ConditionHardeningTests(unittest.TestCase):
    def test_underscore_and_dunder_attributes_blocked(self):
        for expr in ("L.__class__", "L._secret", "R.keys", "L.items", "L.get"):
            with self.subTest(expr=expr):
                with self.assertRaises(ConditionError):
                    Condition.compile(expr)

    def test_pow_resource_bounds(self):
        # 指数超限在求值期收敛为“不接受”，不挂起也不爆内存。
        huge = Condition.compile("L.v ** 100000 > 0")
        self.assertFalse(huge.evaluate({"v": 2}, {}))
        huge_call = Condition.compile("pow(L.v, 200) > 0")
        self.assertFalse(huge_call.evaluate({"v": 2}, {}))
        # 正常范围内的幂仍然可用。
        ok = Condition.compile("pow(L.v, 2) > 10")
        self.assertTrue(ok.evaluate({"v": 4}, {}))
        ok2 = Condition.compile("L.v ** 3 == 8")
        self.assertTrue(ok2.evaluate({"v": 2}, {}))

    def test_non_numeric_pow_rejected_as_pair(self):
        cond = Condition.compile("L.v ** R.v > 0")
        self.assertFalse(cond.evaluate({"v": "x"}, {"v": 2}))


class StatsValidationTests(unittest.TestCase):
    def test_chi_square_input_validation(self):
        with self.assertRaises(ValueError):
            chi_square_uniform([1, 2], [0.5])  # 长度不一致
        with self.assertRaises(ValueError):
            chi_square_uniform([1, 2], [0.7, 0.7])  # 和不为 1
        with self.assertRaises(ValueError):
            chi_square_uniform([1, -1], [0.5, 0.5])  # 负计数
        with self.assertRaises(ValueError):
            chi_square_uniform([], [])

    def test_binomial_input_validation(self):
        with self.assertRaises(ValueError):
            binomial_two_sided_p(5, 4, 0.5)  # k > n
        with self.assertRaises(ValueError):
            binomial_two_sided_p(1, 10, 1.2)  # p 越界
        with self.assertRaises(ValueError):
            binomial_two_sided_p(1, -1, 0.5)
        # 合法调用仍工作。
        self.assertGreater(binomial_two_sided_p(5, 10, 0.5), 0.0)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            SamplerConfig(rejection_attempt_cap=0).validate()
        with self.assertRaises(ValueError):
            SamplerConfig(max_accepted_cache_pairs=0).validate()
        with self.assertRaises(ValueError):
            SamplerConfig(sample_size=0).validate()
        with self.assertRaises(ValueError):
            SamplerConfig(mode="XX").validate()
        with self.assertRaises(ValueError):
            SamplerConfig(max_build_seconds=-1).validate()


class MixedTypeKeySortTests(unittest.TestCase):
    def test_heterogeneous_tuple_keys_sort_and_sample(self):
        keys = [5, "x", (1, "a"), (2, 3)]
        rows_l = [{"k": k, "v": i} for i, k in enumerate(keys)]
        rows_r = [{"k": k, "v": i} for i, k in enumerate(keys)]
        snap = build_snapshot(
            side_from_rows(rows_l, name="l", key="k"),
            side_from_rows(rows_r, name="r", key="k"),
        )
        first = snap.keys()
        # 确定性：再建一次顺序一致，且不抛 TypeError。
        snap2 = build_snapshot(
            side_from_rows(list(rows_l), name="l", key="k"),
            side_from_rows(list(rows_r), name="r", key="k"),
        )
        self.assertEqual(first, snap2.keys())
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=8, mode="WR", rng_seed=3)
        ).build()
        records = sampler.draw_more(8)
        self.assertTrue(all(r["key"] in keys for r in records))


class CheckpointHardeningTests(unittest.TestCase):
    def setUp(self):
        self.snap = _single_key_snapshot("L.v + R.v <= 15", nl=12, nr=12)

    def _sampler(self, **over):
        cfg = dict(sample_size=20, mode="WOR", rng_seed=77)
        cfg.update(over)
        return JoinSampler(self.snap, SamplerConfig(**cfg)).build()

    def test_corrupt_and_truncated_checkpoints_raise_resume_error(self):
        sampler = self._sampler()
        sampler.draw_more(10)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp.json")
            save_checkpoint(sampler, path)
            with open(path, "rb") as fh:
                raw = fh.read()
            for bad in (b"{not json", raw[: len(raw) // 2]):
                with open(path, "wb") as fh:
                    fh.write(bad)
                with self.assertRaises(ResumeError):
                    load_checkpoint(path)

    def test_tampered_structure_raises_resume_error(self):
        sampler = self._sampler()
        sampler.draw_more(10)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp.json")
            save_checkpoint(sampler, path)
            state = load_checkpoint(path)
            state["config"].pop("mode")
            with self.assertRaises(ResumeError):
                JoinSampler.restore(state, self.snap)

    def test_expect_config_mismatch_rejected_equality_accepted(self):
        sampler = self._sampler()
        sampler.draw_more(10)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp.json")
            save_checkpoint(sampler, path)
            good = dict(sampler.config.__dict__)
            restored = restore_sampler(path, self.snap, expect_config=good)
            self.assertEqual(len(restored._records), 10)
            bad = dict(good)
            bad["mode"] = "WR"
            with self.assertRaises(ResumeError):
                restore_sampler(path, self.snap, expect_config=bad)

    def test_alive_cache_never_persisted_and_resume_continues(self):
        # 小缓存（蓄水池路径）+ 极小拒绝 cap，WOR 抽取时强制触顶并构建
        # alive_cache，再确认该运行期派生物不进入检查点。
        sampler = JoinSampler(
            self.snap,
            SamplerConfig(
                sample_size=100, mode="WOR", rng_seed=77,
                max_pairs_per_key=10_000, max_accepted_cache_pairs=8,
                rejection_attempt_cap=4,
            ),
        ).build()
        self.assertTrue(sampler._infos[0].cache_is_reservoir)
        sampler.draw_more(100)
        self.assertIsNotNone(sampler._infos[0].alive_cache)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp.json")
            save_checkpoint(sampler, path)
            with open(path, encoding="utf-8") as fh:
                self.assertNotIn("alive_cache", fh.read())
            restored = JoinSampler.restore(load_checkpoint(path), self.snap)
            self.assertIsNone(restored._infos[0].alive_cache)
            more = restored.draw_more(10)
            self.assertEqual(len(more), 10)
            self.assertEqual(len(restored._records), 110)
            self.assertEqual(
                len({(r["left_row_id"], r["right_row_id"])
                     for r in restored._records}),
                110,
            )

    def test_checkpoint_file_is_strict_json_without_nan(self):
        sampler = self._sampler()
        sampler.draw_more(5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp.json")
            save_checkpoint(sampler, path)
            # 严格 JSON 解析器必须能直接读回（默认 parse 已拒绝 NaN 以外的
            # 非标准记号；这里再显式用 parse_constant 拦截 NaN/Infinity）。
            with open(path, encoding="utf-8") as fh:
                def _reject_nonstandard(constant: str):
                    raise ValueError(f"非法 JSON 常量 {constant}")
                json.load(fh, parse_constant=_reject_nonstandard)


if __name__ == "__main__":
    unittest.main()
