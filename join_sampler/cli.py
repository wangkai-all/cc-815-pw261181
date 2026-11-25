"""命令行入口。

示例：
  python -m join_sampler.cli sample --left l.csv --right r.csv --key k \
      --n 200 --mode WOR --seed 7 --out run.json --checkpoint cp.json
  python -m join_sampler.cli resume --checkpoint cp.json --more 100 \
      --left l.csv --right r.csv --key k --out run2.json
  python -m join_sampler.cli synthetic --out-dir ./demo --n 500 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .checkpoint import save_checkpoint, load_checkpoint, ResumeError
from .conditions import Condition, ConditionError
from .data import load_csv_side, make_synthetic_tables
from .sampler import JoinSampler, SamplerConfig, SamplingError, ResourceLimitExceeded
from .snapshot import build_snapshot


def _json_default(obj: Any):
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return repr(obj)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)


def cmd_sample(args: argparse.Namespace) -> int:
    left = load_csv_side(args.left, name="left", key=args.key)
    right = load_csv_side(args.right, name="right", key=args.key)
    try:
        cond = Condition.compile(args.condition)
    except ConditionError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    snapshot = build_snapshot(left, right, cond)
    cfg = SamplerConfig(
        sample_size=args.n,
        mode=args.mode,
        rng_seed=args.seed,
        max_pairs_per_key=args.max_pairs_per_key,
        estimator_probes=args.estimator_probes,
        max_build_evaluations=args.max_evaluations,
        max_build_seconds=args.max_seconds,
        max_memory_bytes=args.max_memory,
    )
    try:
        sampler = JoinSampler(snapshot, cfg).build()
        sampler.draw_more(args.n)
        result = sampler.result()
    except (SamplingError, ResourceLimitExceeded) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        if isinstance(exc, ResourceLimitExceeded):
            _write_json(args.out, {"error": str(exc), "partial": exc.partial})
        return 3
    _write_json(args.out, result.to_jsonable())
    if args.checkpoint:
        save_checkpoint(sampler, args.checkpoint)
    print(
        f"完成：{len(result.records)} 个关联对 -> {args.out}"
        + (f"，检查点 -> {args.checkpoint}" if args.checkpoint else "")
    )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    try:
        state = load_checkpoint(args.checkpoint)
    except ResumeError as exc:
        # 损坏/截断/结构不兼容的检查点在 load 阶段即被收敛为 ResumeError。
        print(f"[错误] 检查点不可用: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[错误] 无法读取检查点: {exc}", file=sys.stderr)
        return 2
    if args.seed is not None and args.seed != state["rng_seed"]:
        print(
            "[错误] 拒绝恢复：不能更换种子重抽样本（检查点种子 "
            f"{state['rng_seed']}，请求 {args.seed}）",
            file=sys.stderr,
        )
        return 2
    left = load_csv_side(args.left, name="left", key=args.key)
    right = load_csv_side(args.right, name="right", key=args.key)
    cond = Condition.compile(state["condition_expression"])
    snapshot = build_snapshot(left, right, cond)
    try:
        sampler = JoinSampler.restore(state, snapshot)
        sampler.draw_more(args.more)
    except (ResumeError, SamplingError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 3
    result = sampler.result()
    _write_json(args.out, result.to_jsonable())
    save_checkpoint(sampler, args.checkpoint)
    print(f"恢复并追加 {args.more} 个关联对，当前共 {len(result.records)} 对 -> {args.out}")
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    import csv
    import os

    os.makedirs(args.out_dir, exist_ok=True)
    left, right = make_synthetic_tables(rng_seed=args.seed)
    for side, fname in ((left, "left.csv"), (right, "right.csv")):
        with open(os.path.join(args.out_dir, fname), "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(side.columns))
            writer.writeheader()
            for i in range(side.row_count):
                writer.writerow(side.get_row(i))
    print(f"合成数据已写入 {args.out_dir}/left.csv 与 right.csv")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="join_sampler", description="两表关联两级抽样工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="构建快照并抽样")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--condition", default=None, help="成对条件，如 'abs(L.v - R.v) <= 5'")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--mode", choices=["WR", "WOR"], default="WR")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-pairs-per-key", type=int, default=100_000)
    p.add_argument("--estimator-probes", type=int, default=2000)
    p.add_argument("--max-evaluations", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--max-memory", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_sample)

    r = sub.add_parser("resume", help="从检查点恢复并追加抽样")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--left", required=True)
    r.add_argument("--right", required=True)
    r.add_argument("--key", required=True)
    r.add_argument("--more", type=int, required=True)
    r.add_argument("--seed", type=int, default=None, help="仅用于校验；必须与原种子一致")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_resume)

    s = sub.add_parser("synthetic", help="生成演示用合成 CSV")
    s.add_argument("--out-dir", default="./demo_data")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_synthetic)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
