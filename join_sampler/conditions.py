"""成对条件（pairwise condition）的安全求值器。

条件作用于关联后的 *行对*，可以引用两侧列：
    L.col_x - R.col_y <= 10
    (L.a + R.b) / 2 > 5 and not (L.flag == R.flag)

设计要点：
* 基于 ast 的白名单求值，禁止属性逃逸、调用、导入等任意代码；
* L.* / R.* 为两侧当前行，L 与 R 是两个命名空间，不存在隐式回退，
  因此同名列不会互相遮蔽；
* 任一侧引用了不存在的列或值为 None（缺失值/空键侧）时，该对判为不接受；
* 整键没有任何接受行对时，该键权重校正为 0（见 sampler）。
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import Any, Mapping


class ConditionError(ValueError):
    """条件表达式无法编译或求值配置非法。"""


# 仅开放纯函数与纯比较/算术，不开放任何可逃逸沙箱的内建。
_ALLOWED_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sqrt": math.sqrt,
    "pow": None,  # 占位，下面替换为带资源上限的 _safe_pow（P2-5）
}

# pow 安全边界：指数绝对值与整数结果位数双重封顶，防止
# 2 ** 100000000 之类的 CPU/内存耗尽攻击；浮点结果必须有限。
_MAX_POW_EXP = 100
_MAX_INT_BITS = 4096


def _safe_pow(base: Any, exp: Any) -> Any:
    if isinstance(base, bool) or isinstance(exp, bool) or not isinstance(
        base, (int, float)
    ) or not isinstance(exp, (int, float)):
        raise TypeError("pow 仅支持数值参数")
    if abs(exp) > _MAX_POW_EXP:
        raise OverflowError(f"pow 指数超过安全上限 {_MAX_POW_EXP}")
    if isinstance(base, int) and base.bit_length() > _MAX_INT_BITS:
        raise OverflowError("pow 底数位数超过安全上限")
    result = pow(base, exp)
    if isinstance(result, int) and result.bit_length() > _MAX_INT_BITS:
        raise OverflowError("pow 整数结果位数超过安全上限")
    if isinstance(result, float) and not math.isfinite(result):
        raise OverflowError("pow 浮点结果非有限值")
    return result


_ALLOWED_FUNCS["pow"] = _safe_pow

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_BOOL_OPS = {ast.And: all, ast.Or: any}


class _Compiler:
    """把表达式 AST 编译成 (namespace_left, namespace_right) -> bool 的函数。"""

    def __init__(self, expression: str) -> None:
        self.expression = expression
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ConditionError(f"条件表达式语法错误: {exc}") from exc
        self._root = self._compile_node(tree.body)

    # -- 节点编译 -------------------------------------------------------
    def _compile_node(self, node: ast.AST):
        if isinstance(node, ast.BoolOp):
            if type(node.op) not in _BOOL_OPS:
                raise ConditionError("不支持的布尔运算")
            parts = [self._compile_node(v) for v in node.values]
            combiner = _BOOL_OPS[type(node.op)]
            return lambda l, r: combiner(p(l, r) for p in parts)

        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ConditionError(f"不支持的一元运算: {type(node.op).__name__}")
            operand = self._compile_node(node.operand)
            return lambda l, r: op(operand(l, r))

        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ConditionError(f"不支持的二元运算: {type(node.op).__name__}")
            left = self._compile_node(node.left)
            right = self._compile_node(node.right)
            return lambda l, r: op(left(l, r), right(l, r))

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ConditionError("仅支持单段比较 (a < b)，不支持链式比较")
            cmp_op = _CMP_OPS.get(type(node.ops[0]))
            if cmp_op is None:
                raise ConditionError(
                    f"不支持的比较运算: {type(node.ops[0]).__name__}"
                )
            left = self._compile_node(node.left)
            right = self._compile_node(node.comparators[0])
            return lambda l, r: bool(cmp_op(left(l, r), right(l, r)))

        if isinstance(node, ast.IfExp):
            test = self._compile_node(node.test)
            body = self._compile_node(node.body)
            orelse = self._compile_node(node.orelse)
            return lambda l, r: body(l, r) if test(l, r) else orelse(l, r)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ConditionError("仅支持白名单内的普通函数调用")
            fname = node.func.id
            if fname not in _ALLOWED_FUNCS:
                raise ConditionError(f"函数不在白名单中: {fname}")
            if node.keywords:
                raise ConditionError("条件函数不支持关键字参数")
            args = [self._compile_node(a) for a in node.args]
            func = _ALLOWED_FUNCS[fname]
            return lambda l, r: func(*(a(l, r) for a in args))

        if isinstance(node, ast.Name):
            # 仅允许布尔字面量与 None 形式的裸名（None 单独成节点，见下）。
            if node.id in ("True", "False"):
                const = node.id == "True"
                return lambda l, r: const
            raise ConditionError(
                f"未知标识符 {node.id!r}：列必须以 L. 或 R. 为前缀引用"
            )

        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id not in ("L", "R"):
                raise ConditionError("属性访问仅允许 L.<列名> 或 R.<列名>")
            side = node.value.id
            col = node.attr
            self._validate_column_name(side, col)
            return lambda l, r: self._read_column(side, col, l, r)

        if isinstance(node, ast.Constant):
            const = node.value
            if isinstance(const, bool) or const is None or isinstance(const, (int, float, str)):
                return lambda l, r: const
            raise ConditionError(f"不支持的常量类型: {type(const).__name__}")

        raise ConditionError(f"不支持的语法节点: {type(node).__name__}")

    @staticmethod
    def _validate_column_name(side: str, col: str) -> None:
        """编译期拒绝可触碰映射内部/方法的列名（P2-4）。

        * 下划线开头（含双下划线 dunder，如 __class__）一律禁止；
        * 映射协议方法名即使数据行是普通 dict 也可能解析到方法对象，
          造成语义混淆，显式拒绝。
        """
        if col.startswith("_"):
            raise ConditionError(f"列名不允许以下划线开头: {side}.{col}")
        if col in ("keys", "values", "items", "get", "copy", "update", "pop"):
            raise ConditionError(
                f"列名 {col!r} 与映射内建方法冲突，请在数据侧重命名该列"
            )

    @staticmethod
    def _read_column(side: str, col: str, left: Mapping[str, Any], right: Mapping[str, Any]):
        row = left if side == "L" else right
        if col not in row:
            # 运行期缺列：与 None 参与运算同样按“该对不接受”处理，
            # 抛 KeyError 由 evaluate 统一捕获，区别于编译期的 ConditionError。
            raise KeyError(f"{side}.{col}")
        return row[col]

    # -- 对外入口 -------------------------------------------------------
    def evaluate(self, left_row: Mapping[str, Any], right_row: Mapping[str, Any]) -> bool:
        try:
            return bool(self._root(left_row, right_row))
        except ConditionError:
            raise
        except (
            TypeError,
            ValueError,
            KeyError,
            ZeroDivisionError,
            OverflowError,
            ArithmeticError,
            MemoryError,
        ):
            # 类型不兼容、缺列、除零、数值溢出或安全上限触发的内存错误，
            # 一律视为该对不接受（P2-5 把 MemoryError 也收敛到这里）。
            return False


@dataclass(frozen=True)
class Condition:
    """编译后的成对条件。

    expression 保留原始文本，用于快照指纹与检查点校验；
    无条件用 None 表示。
    """

    expression: str | None
    _compiler: _Compiler | None

    @classmethod
    def compile(cls, expression: str | None) -> "Condition":
        if expression is None or not expression.strip():
            return cls(None, None)
        return cls(expression, _Compiler(expression))

    def evaluate(self, left_row, right_row) -> bool:
        if self._compiler is None:
            return True
        return self._compiler.evaluate(left_row, right_row)

    @property
    def is_none(self) -> bool:
        return self._compiler is None

    def __str__(self) -> str:  # pragma: no cover - 调试辅助
        return self.expression or "<无条件>"
