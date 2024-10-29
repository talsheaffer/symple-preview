import math
from functools import lru_cache

from symple.expr.expr_node import ExprNode, ExprNodeType


def can_eval_ints(expr: ExprNode) -> bool:
    def _can_eval_ints(expr: ExprNode) -> bool:
        if expr.type == ExprNodeType.ADD:
            return _can_eval_ints(expr.left) and _can_eval_ints(expr.right)
        elif expr.type == ExprNodeType.MUL:
            return _can_eval_ints(expr.left) and _can_eval_ints(expr.right)
        elif expr.type == ExprNodeType.NEG:
            return _can_eval_ints(expr.left)
        else:
            return expr.type == ExprNodeType.INT

    return (
        _can_eval_ints(expr)
        and expr.type != ExprNodeType.INT 
        and (
            expr.type != ExprNodeType.NEG
            or expr.left.type != ExprNodeType.INT
        )
    )


def eval_ints(expr: ExprNode) -> ExprNode:
    assert can_eval_ints(expr)

    def _eval_ints(expr: ExprNode) -> int:
        if expr.type == ExprNodeType.ADD:
            return _eval_ints(expr.left) + _eval_ints(expr.right)
        elif expr.type == ExprNodeType.MUL:
            return _eval_ints(expr.left) * _eval_ints(expr.right)
        elif expr.type == ExprNodeType.NEG:
            return -_eval_ints(expr.left)
        else:
            return expr.arg

    expr = ExprNode(ExprNodeType.INT, _eval_ints(expr))
    if expr.arg < 0:
        expr.arg = -expr.arg
        expr = ExprNode(ExprNodeType.NEG, left=expr)
    return expr


@lru_cache(maxsize=None)
def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if (n % i) == 0:
            return False
    return True


@lru_cache(maxsize=None)
def _factor_int(n: int) -> int:
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def can_factor_int(expr: ExprNode) -> bool:
    return expr.type == ExprNodeType.INT and not _is_prime(expr.arg) and expr.arg > 1


def factor_int(expr: ExprNode) -> ExprNode:
    assert can_factor_int(expr)
    factors = _factor_int(expr.arg)
    expr = ExprNode(ExprNodeType.INT, factors[0])
    for factor in factors[1:]:
        expr = ExprNode(ExprNodeType.MUL, left=expr, right=ExprNode(ExprNodeType.INT, factor))
    return expr
