from __future__ import annotations

import ast
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalcError(ValueError):
    """Raised for invalid syntax or anything outside the whitelist."""


def calc(expr: str) -> float:
    """Evaluate a plain arithmetic expression. No names, calls, or imports."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"Invalid syntax: {expr!r}") from exc
    return _eval(tree.body)


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        # bool is a subclass of int — reject it so True/False can't sneak in.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError(f"Only numbers allowed, got {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalcError(f"Operator {type(node.op).__name__} not allowed")
        return op(_eval(node.left), _eval(node.right))

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalcError(f"Unary op {type(node.op).__name__} not allowed")
        return op(_eval(node.operand))

    raise CalcError(f"Disallowed expression: {type(node).__name__}")