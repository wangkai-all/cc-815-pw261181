"""检查点：保存/恢复抽样器随机状态与已选行对。

设计承诺：
* 检查点内含 random.Random 的完整内部状态与全部已抽记录，恢复后
  继续抽样等价于从未中断；
* 检查点绑定快照指纹与种子，恢复到不同快照会被拒绝；
* 不支持「换一个种子把同一批样本重抽一遍」——种子属于配置的一部分，
  恢复时若传入不同 rng_seed 直接报错（ResumeError），新抽样请另建
  新运行。
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .sampler import JoinSampler, JoinSnapshot


class ResumeError(RuntimeError):
    """检查点与当前快照/配置不兼容，无法恢复。"""


def encode_key(key: Any) -> Any:
    """把键编码成 JSON 可表示的带标签结构（支持嵌套 tuple）。"""
    if key is None or isinstance(key, bool):
        return ["b", key]
    if isinstance(key, int):
        return ["i", key]
    if isinstance(key, float):
        return ["f", key]
    if isinstance(key, str):
        return ["s", key]
    if isinstance(key, tuple):
        return ["t", [encode_key(k) for k in key]]
    raise ResumeError(
        f"键类型 {type(key).__name__} 无法写入检查点；"
        "请使用 int/float/str/bool/None/tuple 类型的键"
    )


def decode_key(payload: Any) -> Any:
    tag, value = payload
    if tag in ("i", "f", "s", "b"):
        return value
    if tag == "t":
        return tuple(decode_key(k) for k in value)
    raise ResumeError(f"检查点中存在未知键标签: {tag!r}")


def _encode_structure(obj: Any) -> Any:
    """递归给所有 dict 中名为 'key' 的字段及 key_order 加编码。"""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "key":
                out["key_enc"] = encode_key(v)
            elif k == "key_order":
                out["key_order_enc"] = [encode_key(x) for x in v]
            elif k == "zero_result_keys":
                out["zero_result_keys_enc"] = [encode_key(x) for x in v]
            elif k in ("key_selection_counts",) and isinstance(v, dict):
                out[k] = [
                    {"key_enc": encode_key(kk), "count": vv} for kk, vv in v.items()
                ]
            else:
                out[k] = _encode_structure(v)
        return out
    if isinstance(obj, list):
        return [_encode_structure(x) for x in obj]
    if isinstance(obj, tuple):
        return ["__tuple__", [_encode_structure(x) for x in obj]]
    return obj


def _decode_structure(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "key_enc":
                out["key"] = decode_key(v)
            elif k == "key_order_enc":
                out["key_order"] = [decode_key(x) for x in v]
            elif k == "zero_result_keys_enc":
                out["zero_result_keys"] = [decode_key(x) for x in v]
            elif k == "key_selection_counts" and isinstance(v, list):
                out[k] = {decode_key(item["key_enc"]): item["count"] for item in v}
            else:
                out[k] = _decode_structure(v)
        return out
    if isinstance(obj, list):
        if len(obj) == 2 and obj[0] == "__tuple__":
            return tuple(_decode_structure(x) for x in obj[1])
        return [_decode_structure(x) for x in obj]
    return obj


def save_checkpoint(sampler: "JoinSampler", path: str) -> str:
    state = sampler.export_state()
    encoded = _encode_structure(state)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        # allow_nan=False：NaN/Infinity 不是合法 JSON，静默写入会让其他
        # 解析器或严格模式读不回来（P3-1）。
        json.dump(
            encoded, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        fh.write("\n")
    os.replace(tmp, path)
    return os.path.abspath(path)


def load_checkpoint(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            encoded = json.load(fh)
        return _decode_structure(encoded)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        # 损坏/截断/结构不符的文件统一收敛为 ResumeError，调用方无需感知
        # 底层 JSON 与解码细节（P3-1）。decode_key 自身抛的 ResumeError
        # （RuntimeError 子类）不在此元组内，会原样向上传播。
        raise ResumeError(f"检查点文件已损坏或结构不兼容: {exc}") from exc
    except FileNotFoundError:
        raise


def _configs_equal(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """返回两个配置字典中取值不一致的字段名（只比较双方共有键）。"""
    diffs: list[str] = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            diffs.append(key)
    return diffs


def restore_sampler(
    path: str,
    snapshot: "JoinSnapshot",
    expect_config: dict[str, Any] | None = None,
) -> "JoinSampler":
    """便捷入口：读检查点并在给定快照上恢复抽样器。

    始终校验：快照指纹、内部种子一致性、配置字段健全性（在
    JoinSampler.restore 内）。若提供 ``expect_config``，再逐项比对完整
    配置，任一字段不同即拒绝恢复（P3-2），防止用同一份检查点在改变
    模式/样本量/枚举上限等参数后继续抽样。
    """
    from .sampler import JoinSampler

    state = load_checkpoint(path)
    if state.get("rng_seed") != state["config"].get("rng_seed"):
        raise ResumeError("检查点内部种子字段不一致，文件可能已损坏")
    if expect_config is not None:
        diffs = _configs_equal(state["config"], dict(expect_config))
        if diffs:
            raise ResumeError(
                "当前配置与检查点记录不一致，拒绝恢复（差异字段: "
                f"{', '.join(diffs)}）。需要不同参数请新建运行，"
                "不允许在恢复时改配置或换种子重抽。"
            )
    return JoinSampler.restore(state, snapshot)
