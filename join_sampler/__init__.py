"""join_sampler —— 两表按键关联的两级均匀抽样工具。

核心约束（见 README）：
* 不先在各表内随机抽行再关联（那样会破坏关联对的等概率性）；
* 不物化完整关联结果（可能达数亿乃至上百亿行）；
* 计数与取样来自同一不可变快照；
* 带成对条件时按各键接受率校正权重，不沿用无条件概率。
"""

from .store import SegmentedColumn, SideData
from .snapshot import JoinSnapshot, build_snapshot
from .conditions import Condition, ConditionError
from .sampler import (
    SamplerConfig,
    JoinSampler,
    SamplingResult,
    ResourceLimitExceeded,
)
from .checkpoint import save_checkpoint, load_checkpoint, ResumeError

__all__ = [
    "SegmentedColumn",
    "SideData",
    "JoinSnapshot",
    "build_snapshot",
    "Condition",
    "ConditionError",
    "SamplerConfig",
    "JoinSampler",
    "SamplingResult",
    "ResourceLimitExceeded",
    "save_checkpoint",
    "load_checkpoint",
    "ResumeError",
]

__version__ = "1.0.0"
