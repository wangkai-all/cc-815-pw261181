"""小表全枚举检验：每个合法关联对必须有相同的入选机会。

做法：构造所有关联对均可枚举的小表，反复抽样后对
* 行对层面的经验分布与理论均匀分布做卡方检验；
* 单个行对的命中比例做二项精确检验。
同时覆盖 WR/WOR、重复源行的独立身份、热点键、空值键。
"""

import math
import unittest
from collections import Counter

from join_sampler.conditions import Condition
from join_sampler.data import side_from_rows
from join_sampler.sampler import JoinSampler, SamplerConfig
from join_sampler.snapshot import build_snapshot
from join_sampler.stats import chi_square_uniform, binomial_two_sided_p


def build_small_snapshot(condition=None):
    # 键分布刻意不均：键 1 是热点；键 99 只在左表、键 88 只在右表；
    # 另有 None 空键行。两侧都有完全重复的行，检验独立行身份。
    left_rows = [
        {"k": 1, "v": 1}, {"k": 1, "v": 2}, {"k": 1, "v": 2},  # 后两行完全重复
        {"k": 2, "v": 5},
        {"k": 3, "v": 9}, {"k": 3, "v": 10},
        {"k": 99, "v": 1},
        {"k": None, "v": 7},
    ]
    right_rows = [
        {"k": 1, "v": 2}, {"k": 1, "v": 2}, {"k": 1, "v": 8},
        {"k": 2, "v": 6}, {"k": 2, "v": 7},
        {"k": 3, "v": 11},
        {"k": 88, "v": 3},
        {"k": None, "v": 7},
    ]
    left = side_from_rows(left_rows, name="left", key="k")
    right = side_from_rows(right_rows, name="right", key="k")
    return build_snapshot(left, right, Condition.compile(condition))


class UniformProbabilityTests(unittest.TestCase):
    def test_unconditioned_wr_each_pair_equal_probability(self):
        snap = build_small_snapshot()
        total_pairs = snap.total_pair_count
        self.assertEqual(total_pairs, 3 * 3 + 1 * 2 + 2 * 1)  # 9+2+2 = 13
        n_trials = 26_000
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=n_trials, mode="WR", rng_seed=1234)
        ).build()
        records = sampler.draw_more(n_trials)

        pair_counts = Counter((r["left_row_id"], r["right_row_id"]) for r in records)
        self.assertEqual(len(pair_counts), total_pairs)  # 每个对都应被抽到过
        observed = [pair_counts.get(self._pair_by_index(snap, i), 0)
                    for i in range(total_pairs)]
        expected = [1.0 / total_pairs] * total_pairs
        stat, p_value, groups = chi_square_uniform(observed, expected)
        self.assertGreater(p_value, 0.01, f"卡方检验异常: stat={stat:.2f}, p={p_value:.4f}")

        # 二项精确检验：随机抽 3 个对，其命中比例不应显著偏离 1/N。
        for pair, count in list(pair_counts.items())[:3]:
            p_binom = binomial_two_sided_p(count, n_trials, 1.0 / total_pairs)
            self.assertGreater(p_binom, 0.01, f"对 {pair} 二项检验失败 p={p_binom:.4f}")

        # 记录里的联合入选概率必须等于 1/总对数（无条件均匀的直接证据）。
        for r in records[:20]:
            self.assertAlmostEqual(r["p_joint"], 1.0 / total_pairs, places=10)

    def test_unconditioned_wor_exhausts_every_pair_once(self):
        snap = build_small_snapshot()
        total_pairs = snap.total_pair_count
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=total_pairs, mode="WOR", rng_seed=4321)
        ).build()
        records = sampler.draw_more(total_pairs)
        pairs = [(r["left_row_id"], r["right_row_id"]) for r in records]
        self.assertEqual(len(pairs), total_pairs)
        self.assertEqual(len(set(pairs)), total_pairs)  # 无重复
        self.assertEqual(set(pairs), set(self._all_pairs(snap)))
        # 再抽一个必须报容量错误
        with self.assertRaisesRegex(Exception, "WOR 容量不足"):
            sampler.draw_more(1)

    def test_duplicate_rows_keep_independent_identity(self):
        # 左表行 1、2 内容完全相同，但行号不同；两者各自形成的对都要出现。
        snap = build_small_snapshot()
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=8000, mode="WR", rng_seed=55)
        ).build()
        records = sampler.draw_more(8000)
        left_ids = {r["left_row_id"] for r in records if r["key"] == 1}
        self.assertEqual(left_ids, {0, 1, 2})  # 重复内容的三行身份独立
        # 内容相同的两行命中频次应接近（差异在 3 个标准差内）。
        c1 = sum(1 for r in records if r["left_row_id"] == 1)
        c2 = sum(1 for r in records if r["left_row_id"] == 2)
        self.assertLess(abs(c1 - c2), 3 * math.sqrt(8000 / 13))

    def test_null_keys_never_join(self):
        snap = build_small_snapshot()
        self.assertEqual(snap.null_left_rows, 1)
        self.assertEqual(snap.null_right_rows, 1)
        self.assertNotIn(None, snap.keys())
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=500, mode="WR", rng_seed=9)
        ).build()
        records = sampler.draw_more(500)
        self.assertTrue(all(r["key"] is not None for r in records))

    def test_nan_key_treated_as_null(self):
        nan = float("nan")
        left = side_from_rows(
            [{"k": 1, "v": 1}, {"k": nan, "v": 2}, {"k": None, "v": 3}],
            name="left", key="k",
        )
        right = side_from_rows(
            [{"k": 1, "v": 9}, {"k": nan, "v": 8}], name="right", key="k"
        )
        snap = build_snapshot(left, right)
        self.assertEqual(snap.null_left_rows, 2)  # NaN 与 None 都计入空键
        self.assertEqual(snap.null_right_rows, 1)
        self.assertEqual(snap.keys(), [1])
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=50, mode="WR", rng_seed=4)
        ).build()
        records = sampler.draw_more(50)
        self.assertTrue(all(r["key"] == 1 for r in records))

    def test_hot_key_share_matches_pair_count_share(self):
        snap = build_small_snapshot()
        # 键 1 有 9 对，占 13 对中的 9/13。
        n = 13_000
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=n, mode="WR", rng_seed=77)
        ).build()
        records = sampler.draw_more(n)
        hot = sum(1 for r in records if r["key"] == 1)
        p = binomial_two_sided_p(hot, n, 9.0 / 13.0)
        self.assertGreater(p, 0.01, f"热点键占比偏离: {hot}/{n}, p={p:.4f}")

    @staticmethod
    def _all_pairs(snap):
        out = []
        for k in snap.keys():
            for lid in snap.left_row_ids(k):
                for rid in snap.right_row_ids(k):
                    out.append((lid, rid))
        return out

    @staticmethod
    def _pair_by_index(snap, idx):
        return UniformProbabilityTests._all_pairs(snap)[idx]


if __name__ == "__main__":
    unittest.main()
