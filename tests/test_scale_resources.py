"""大表检验：不物化关联结果与资源上限。

场景：两侧各约 20 万行、500 个键 + 热点键，全量关联对达数亿级。
断言：
* 完整关联结果从不物化（materialized_join_rows 恒为 0，
  且进程中不存在与总对数同规模的容器）；
* 内存随表规模线性、与关联对数（乘积规模）脱钩；
* 求值次数 / 时间 / 内存三类资源上限均能中止构建并返回部分统计；
* 抽样结果仍满足基本分布性质。
"""

import gc
import unittest

from join_sampler.conditions import Condition
from join_sampler.data import make_synthetic_tables
from join_sampler.sampler import JoinSampler, SamplerConfig, ResourceLimitExceeded
from join_sampler.snapshot import build_snapshot
from join_sampler.stats import chi_square_uniform
from collections import Counter


class ScaleAndResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.left, cls.right = make_synthetic_tables(
            left_rows=200_000, right_rows=200_000, distinct_keys=500,
            hot_key_multiplier=300, null_key_fraction=0.01, rng_seed=2026,
        )
        cls.snap = build_snapshot(
            cls.left, cls.right, Condition.compile("abs(L.v - R.v) <= 50")
        )

    def test_join_output_never_materialized(self):
        total_pairs = self.snap.total_pair_count
        self.assertGreater(total_pairs, 80_000_000)  # 数亿级关联对

        sampler = JoinSampler(
            self.snap,
            SamplerConfig(sample_size=1000, mode="WOR", rng_seed=3,
                          max_pairs_per_key=50_000, estimator_probes=1000,
                          max_accepted_cache_pairs=50_000),
        ).build()
        records = sampler.draw_more(1000)
        report = sampler.memory_report()

        self.assertEqual(report["materialized_join_rows"], 0)
        self.assertLessEqual(report["accepted_pairs_cached"], 50_000)
        # 内存上限：数亿关联对若按每对至少 2 个 Python int 估算 > 5GB，
        # 实际结构内存应在百 MB 量级以内。
        self.assertLess(report["store_bytes"] + report["sampler_bytes"], 400_000_000)
        self.assertEqual(len(records), 1000)
        # WOR 无重复对。
        pairs = {(r["left_row_id"], r["right_row_id"]) for r in records}
        self.assertEqual(len(pairs), 1000)

    def test_memory_scales_with_rows_not_pair_count(self):
        # 半规模表：关联对约为 1/4，但表内存应约为 1/2。
        l2, r2 = make_synthetic_tables(
            left_rows=100_000, right_rows=100_000, distinct_keys=500,
            hot_key_multiplier=300, null_key_fraction=0.01, rng_seed=2026,
        )
        snap2 = build_snapshot(l2, r2, Condition.compile("abs(L.v - R.v) <= 50"))
        s1 = JoinSampler(self.snap, SamplerConfig(sample_size=10, rng_seed=1,
                                                  max_pairs_per_key=50_000)).build()
        s2 = JoinSampler(snap2, SamplerConfig(sample_size=10, rng_seed=1,
                                              max_pairs_per_key=50_000)).build()
        gc.collect()
        m1 = s1.memory_report()["store_bytes"]
        m2 = s2.memory_report()["store_bytes"]
        # 半规模存储不应超过全规模的 70%（留分段/字典开销余量）。
        self.assertLess(m2, m1 * 0.70)

    def test_evaluation_budget_aborts_build(self):
        cfg = SamplerConfig(
            sample_size=10, rng_seed=4, max_pairs_per_key=10_000_000,
            max_build_evaluations=5_000,
        )
        sampler = JoinSampler(self.snap, cfg)
        with self.assertRaises(ResourceLimitExceeded) as ctx:
            sampler.build()
        self.assertEqual(ctx.exception.limit, "max_build_evaluations")
        self.assertGreater(ctx.exception.used, 5_000)
        self.assertIn("keys_processed", ctx.exception.partial)

    def test_time_budget_aborts_build(self):
        cfg = SamplerConfig(
            sample_size=10, rng_seed=5, max_pairs_per_key=10_000_000,
            max_build_seconds=0.001,
        )
        with self.assertRaises(ResourceLimitExceeded) as ctx:
            JoinSampler(self.snap, cfg).build()
        self.assertEqual(ctx.exception.limit, "max_build_seconds")

    def test_memory_budget_aborts_build(self):
        cfg = SamplerConfig(
            sample_size=10, rng_seed=6, max_pairs_per_key=10_000_000,
            max_memory_bytes=1024,  # 1KB，必然在首个键后触发
        )
        with self.assertRaises(ResourceLimitExceeded) as ctx:
            JoinSampler(self.snap, cfg).build()
        self.assertEqual(ctx.exception.limit, "max_memory_bytes")

    def test_large_scale_distribution_sanity(self):
        # 无条件大表抽样：键命中占比应贴合 pair_count 占比。
        snap = build_snapshot(self.left, self.right)
        n = 8_000
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=n, mode="WR", rng_seed=88)
        ).build()
        records = sampler.draw_more(n)
        counts = Counter(r["key"] for r in records)
        # 取前 12 个高权重键做卡方。
        top = sorted(snap.keys(), key=lambda k: -snap.stat(k).pair_count)[:12]
        total_weight = sum(snap.stat(k).pair_count for k in top)
        observed = [counts.get(k, 0) for k in top]
        # 条件于「落入这些键」的概率。
        in_top = sum(observed)
        probs = [snap.stat(k).pair_count / total_weight for k in top]
        _, p_value, _ = chi_square_uniform(observed, probs)
        self.assertGreater(in_top, n * 0.5)  # 热点键理应占据多数
        self.assertGreater(p_value, 0.01, f"大表键分布异常 p={p_value:.4f}")


if __name__ == "__main__":
    unittest.main()
