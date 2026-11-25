"""快照不可变性与检查点恢复检验。

* 计数与取样来自同一快照：快照构建后再向两侧追加段，已建抽样器结果不变；
* 检查点恢复后随机状态延续，分段抽样与一次连续抽样逐条一致；
* 拒绝换种子重抽、拒绝异快照恢复；
* 检查点文件可落盘再读回。
"""

import os
import tempfile
import unittest

from join_sampler.checkpoint import save_checkpoint, load_checkpoint, ResumeError
from join_sampler.conditions import Condition
from join_sampler.data import make_synthetic_tables
from join_sampler.sampler import JoinSampler, SamplerConfig
from join_sampler.snapshot import build_snapshot


class SnapshotImmutabilityTests(unittest.TestCase):
    def test_snapshot_stable_under_later_appends(self):
        left, right = make_synthetic_tables(
            left_rows=3000, right_rows=3000, distinct_keys=30, rng_seed=101
        )
        snapshot = build_snapshot(left, right, Condition.compile("L.v <= R.v"))
        before = snapshot.describe()
        sampler = JoinSampler(
            snapshot, SamplerConfig(sample_size=500, mode="WR", rng_seed=909)
        ).build()
        records_before = sampler.draw_more(500)

        # 快照建立后再追加新段（含已存在键与新键）。
        left.append_segment([{"k": 0, "v": 1, "src": "new-L"},
                             {"k": 777, "v": 2, "src": "new-L2"}])
        right.append_segment([{"k": 0, "v": 9, "src": "new-R"},
                              {"k": 888, "v": 2, "src": "new-R2"}])

        after = snapshot.describe()
        self.assertEqual(before, after)  # 快照视图不变
        self.assertEqual(snapshot.fingerprint, before["fingerprint"])
        more = sampler.draw_more(500)
        # 新追加的行号（3000、3001）不得出现在基于旧快照的样本里。
        for r in more:
            self.assertLess(r["left_row_id"], 3000)
            self.assertLess(r["right_row_id"], 3000)
        self.assertEqual(len(records_before) + len(more), 1000)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.left, self.right = make_synthetic_tables(
            left_rows=4000, right_rows=4000, distinct_keys=40, rng_seed=202
        )
        self.snapshot = build_snapshot(
            self.left, self.right, Condition.compile("abs(L.v - R.v) <= 200")
        )

    def test_resume_matches_uninterrupted_run_wr(self):
        cfg = SamplerConfig(sample_size=120, mode="WR", rng_seed=31337)
        reference = JoinSampler(self.snapshot, cfg).build()
        ref_records = reference.draw_more(120)

        split = JoinSampler(
            self.snapshot, SamplerConfig(sample_size=40, mode="WR", rng_seed=31337)
        ).build()
        split.draw_more(40)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp.json")
            save_checkpoint(split, path)
            self.assertTrue(os.path.getsize(path) > 0)
            state = load_checkpoint(path)
            restored = JoinSampler.restore(state, self.snapshot)
            restored.draw_more(80)

        ref_pairs = [(r["left_row_id"], r["right_row_id"])
                     for r in ref_records]
        split_pairs = [(r["left_row_id"], r["right_row_id"])
                       for r in restored._records]
        self.assertEqual(ref_pairs, split_pairs)
        # draw_index 在恢复后延续而非重置。
        self.assertEqual(restored._records[40]["draw_index"], 40)

    def test_resume_matches_uninterrupted_run_wor(self):
        cfg = SamplerConfig(sample_size=150, mode="WOR", rng_seed=4242,
                            estimator_probes=500, max_pairs_per_key=2000)
        reference = JoinSampler(self.snapshot, cfg).build()
        reference.draw_more(150)

        split = JoinSampler(
            self.snapshot, SamplerConfig(sample_size=60, mode="WOR", rng_seed=4242,
                                         estimator_probes=500, max_pairs_per_key=2000)
        ).build()
        split.draw_more(60)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp_wor.json")
            save_checkpoint(split, path)
            restored = JoinSampler.restore(load_checkpoint(path), self.snapshot)
            restored.draw_more(90)

        self.assertEqual(
            [(r["left_row_id"], r["right_row_id"]) for r in reference._records],
            [(r["left_row_id"], r["right_row_id"]) for r in restored._records],
        )
        # 全局限重集合也恢复了。
        self.assertEqual(len(restored._drawn_pairs), 150)

    def test_reject_foreign_snapshot(self):
        split = JoinSampler(
            self.snapshot, SamplerConfig(sample_size=20, mode="WR", rng_seed=1)
        ).build()
        split.draw_more(20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp.json")
            save_checkpoint(split, path)
            state = load_checkpoint(path)

        other_left, other_right = make_synthetic_tables(
            left_rows=4000, right_rows=4000, distinct_keys=40, rng_seed=999
        )
        other_snap = build_snapshot(
            other_left, other_right, Condition.compile("abs(L.v - R.v) <= 200")
        )
        with self.assertRaises(ResumeError):
            JoinSampler.restore(state, other_snap)

    def test_reject_seed_change(self):
        split = JoinSampler(
            self.snapshot, SamplerConfig(sample_size=20, mode="WR", rng_seed=12345)
        ).build()
        split.draw_more(20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp.json")
            save_checkpoint(split, path)
            state = load_checkpoint(path)
        # 篡改种子必须被识别（不允许换种子重挑样本）。
        state["rng_seed"] = 67890
        with self.assertRaises(ResumeError):
            JoinSampler.restore(state, self.snapshot)

    def test_checkpoint_roundtrip_with_tuple_keys(self):
        rows_l = [{"k": (1, "a"), "v": i} for i in range(5)]
        rows_r = [{"k": (1, "a"), "v": i} for i in range(4)]
        from join_sampler.data import side_from_rows

        snap = build_snapshot(
            side_from_rows(rows_l, name="l", key="k"),
            side_from_rows(rows_r, name="r", key="k"),
        )
        sampler = JoinSampler(
            snap, SamplerConfig(sample_size=10, mode="WR", rng_seed=2)
        ).build()
        sampler.draw_more(10)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cp_tuple.json")
            save_checkpoint(sampler, path)
            restored = JoinSampler.restore(load_checkpoint(path), snap)
        self.assertEqual(restored._infos[0].key, (1, "a"))
        self.assertEqual(len(restored._records), 10)


if __name__ == "__main__":
    unittest.main()
