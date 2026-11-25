"""成对条件下的接受率校正检验。

* 精确枚举键：权重 = pair_count × 接受率，条件后每个 *接受对* 仍等概率；
* 整键零结果：权重为 0，永不被抽中；
* 估计热点键：分布与精确枚举参考分布做卡方对比；
* 条件求值沙箱：危险表达式被拒绝。
"""

import unittest
from collections import Counter

from join_sampler.conditions import Condition, ConditionError
from join_sampler.data import side_from_rows
from join_sampler.sampler import JoinSampler, SamplerConfig
from join_sampler.snapshot import build_snapshot
from join_sampler.stats import chi_square_uniform


def conditioned_snapshot(expression):
    # 每个键的行对规模刻意设计：键 10 为可全枚举的小键，
    # 键 20 为大热点键（走估计路径），键 30 的条件使整键零结果。
    left, right = [], []
    for i in range(12):
        left.append({"k": 10, "v": i})
    for i in range(10):
        right.append({"k": 10, "v": i * 2})
    for i in range(400):
        left.append({"k": 20, "v": i})
    for i in range(400):
        right.append({"k": 20, "v": (i * 7) % 400})
    for i in range(6):
        left.append({"k": 30, "v": i})
    for i in range(5):
        right.append({"k": 30, "v": 1000 + i})
    lside = side_from_rows(left, name="left", key="k")
    rside = side_from_rows(right, name="right", key="k")
    return build_snapshot(lside, rside, Condition.compile(expression))


class ConditionCorrectionTests(unittest.TestCase):
    def test_exact_key_accepted_pairs_uniform(self):
        snap = conditioned_snapshot("L.v + R.v <= 12")
        sampler = JoinSampler(
            snap,
            SamplerConfig(sample_size=12_000, mode="WR", rng_seed=2024,
                          max_pairs_per_key=10_000),  # 键 10 的 120 对全枚举
        ).build()
        records = sampler.draw_more(12_000)

        # 手工枚举键 10 的接受对作为基准。
        accepted = []
        for lid in snap.left_row_ids(10):
            for rid in snap.right_row_ids(10):
                if snap.condition.evaluate(snap.get_left_row(lid), snap.get_right_row(rid)):
                    accepted.append((lid, rid))
        self.assertGreater(len(accepted), 0)

        key10 = [r for r in records if r["key"] == 10]
        counts = Counter((r["left_row_id"], r["right_row_id"]) for r in key10)
        observed = [counts.get(p, 0) for p in accepted]
        probs = [1.0 / len(accepted)] * len(accepted)
        _, p_value, _ = chi_square_uniform(observed, probs)
        self.assertGreater(p_value, 0.01, f"条件后接受对分布不均匀 p={p_value:.4f}")

        # 构建信息层面的精确接受率。
        info10 = sampler._infos[sampler._index[10]]
        self.assertFalse(info10.rate_is_estimate)
        self.assertEqual(info10.accepted_count, len(accepted))
        self.assertAlmostEqual(info10.weight, float(len(accepted)))
        for r in key10[:10]:
            self.assertTrue(r["condition_passed"])
            self.assertAlmostEqual(r["p_select_pair_within_key"], 1.0 / len(accepted), places=8)

    def test_whole_key_zero_result_never_selected(self):
        # 键 30 右行 v>=1000、左行 v<=5；条件 R.v < 500 对键 30 恒假，
        # 而键 10、20 仍有接受对。
        snap = conditioned_snapshot("R.v < 500")
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=4000, mode="WR", rng_seed=7,
                                max_pairs_per_key=10_000)
        ).build()
        info30 = sampler._infos[sampler._index[30]]
        self.assertEqual(info30.acceptance_rate, 0.0)
        self.assertEqual(info30.weight, 0.0)
        records = sampler.draw_more(4000)
        self.assertTrue(all(r["key"] != 30 for r in records))
        self.assertIn(30, sampler._build_stats["zero_result_keys"])

        # 若所有键都零结果，构建后抽样应清晰报错而不是死循环。
        from join_sampler.sampler import SamplingError
        dead = conditioned_snapshot("L.v > 100000")
        dead_sampler = JoinSampler(
            dead, SamplerConfig(sample_size=10, mode="WR", rng_seed=1,
                                max_pairs_per_key=10_000)
        ).build()
        with self.assertRaises(SamplingError):
            dead_sampler.draw_more(10)

    def test_estimator_hotkey_matches_exact_reference_distribution(self):
        expression = "abs(L.v - R.v) <= 20"
        snap = conditioned_snapshot(expression)

        # 参考：把枚举上限调到极大，键 20 也全枚举。
        ref = JoinSampler(
            snap, SamplerConfig(sample_size=20_000, mode="WR", rng_seed=11,
                                max_pairs_per_key=10_000_000, estimator_probes=5000)
        ).build()
        ref_records = ref.draw_more(20_000)
        ref_info20 = ref._infos[ref._index[20]]
        self.assertFalse(ref_info20.rate_is_estimate)

        # 被测：键 20 走探针估计路径。
        est = JoinSampler(
            snap, SamplerConfig(sample_size=20_000, mode="WR", rng_seed=12,
                                max_pairs_per_key=10_000, estimator_probes=3000)
        ).build()
        est_info20 = est._infos[est._index[20]]
        self.assertTrue(est_info20.rate_is_estimate)
        # 估计接受率应落在精确值的合理区间内（3 个二项标准差）。
        import math
        p = ref_info20.acceptance_rate
        se = math.sqrt(p * (1 - p) / 3000)
        self.assertLess(abs(est_info20.acceptance_rate - p), 3 * se + 1e-12)
        est_records = est.draw_more(20_000)

        # 键层面：两次运行的键占比对比。
        ref_keys = Counter(r["key"] for r in ref_records)
        est_keys = Counter(r["key"] for r in est_records)
        keys = sorted(ref_keys)
        obs = [est_keys[k] for k in keys]
        probs = [ref_keys[k] / len(ref_records) for k in keys]
        _, p_value, _ = chi_square_uniform(obs, probs)
        self.assertGreater(p_value, 0.01, f"估计路径键分布偏离参考 p={p_value:.4f}")

    def test_wor_with_condition_no_duplicate_accepted_pairs(self):
        snap = conditioned_snapshot("L.v + R.v <= 200")
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=500, mode="WOR", rng_seed=31,
                                max_pairs_per_key=10_000, estimator_probes=1000)
        ).build()
        records = sampler.draw_more(500)
        pairs = [(r["left_row_id"], r["right_row_id"]) for r in records]
        self.assertEqual(len(set(pairs)), 500)
        self.assertTrue(all(r["condition_passed"] for r in records))

    def test_every_accepted_pair_has_identical_joint_probability(self):
        """带条件时校正接受率后，所有接受对的联合入选概率严格相等 (1/M)。"""
        snap = conditioned_snapshot("abs(L.v - R.v) <= 7")
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=30_000, mode="WR", rng_seed=314,
                                max_pairs_per_key=10_000_000)
        ).build()
        accepted = []
        for k in snap.keys():
            for lid in snap.left_row_ids(k):
                for rid in snap.right_row_ids(k):
                    if snap.condition.evaluate(
                        snap.get_left_row(lid), snap.get_right_row(rid)
                    ):
                        accepted.append((lid, rid))
        m = len(accepted)
        records = sampler.draw_more(30_000)
        counts = Counter((r["left_row_id"], r["right_row_id"]) for r in records)
        observed = [counts.get(p, 0) for p in accepted]
        _, p_value, _ = chi_square_uniform(observed, [1.0 / m] * m)
        self.assertGreater(p_value, 0.01)
        # 记录里声明的联合概率必须等于 1/M，而非无条件的 1/总对数。
        total_pairs = snap.total_pair_count
        self.assertNotEqual(m, total_pairs)
        for r in records[:30]:
            self.assertAlmostEqual(r["p_joint"], 1.0 / m, places=12)
            self.assertNotAlmostEqual(r["p_joint"], 1.0 / total_pairs, places=12)


class ConditionSandboxTests(unittest.TestCase):
    def test_dangerous_expressions_rejected(self):
        bad = [
            "__import__('os').system('echo x')",
            "open('/etc/passwd')",
            "(lambda: 1)()",
            "L.v.__class__.__bases__",
            "L.v if True else R.v",  # 这个合法，下面单独确认
        ]
        for expr in bad[:-1]:
            with self.assertRaises(ConditionError):
                Condition.compile(expr)

    def test_safe_expression_and_missing_value(self):
        cond = Condition.compile("abs(L.v - R.v) <= 2 and L.flag != R.flag")
        self.assertTrue(cond.evaluate({"v": 1, "flag": True}, {"v": 3, "flag": False}))
        self.assertFalse(cond.evaluate({"v": 1}, {"v": 3}))  # 缺列 -> 不接受
        # 除零等数值异常按不接受处理而非抛出。
        cond2 = Condition.compile("L.a / R.b > 1")
        self.assertFalse(cond2.evaluate({"a": 1}, {"b": 0}))
        # 三元表达式合法，且分支数值可以参与后续比较。
        cond3 = Condition.compile("(L.v if L.v > R.v else R.v) > 3")
        self.assertTrue(cond3.evaluate({"v": 5}, {"v": 2}))
        self.assertFalse(cond3.evaluate({"v": 1}, {"v": 2}))


if __name__ == "__main__":
    unittest.main()
