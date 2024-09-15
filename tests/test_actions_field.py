import pytest
import sympy as sp

from symple.expr import ExprNode, af


def test_commute():
    expr = sp.sympify("x + y", evaluate=False)
    node = ExprNode.from_sympy(expr)
    commuted_node = af.commute(node)
    assert repr(commuted_node) == "(y + x)"

def test_associate_b():
    expr = sp.sympify("x + (y + z)", evaluate=False)
    node = ExprNode.from_sympy(expr)
    associated_node = af.associate_b(node)
    assert repr(associated_node) == "((x + y) + z)"

def test_distribute_b():
    expr = sp.sympify("x * (y + z)", evaluate=False)
    node = ExprNode.from_sympy(expr)
    distributed_node = af.distribute_b(node)
    assert repr(distributed_node) == "((x * y) + (x * z))"

def test_undistribute_b():
    expr = sp.sympify("(x * y) + (x * z)", evaluate=False)
    node = ExprNode.from_sympy(expr)
    undistributed_node = af.undistribute_b(node)
    assert repr(undistributed_node) == "(x * (y + z))"

@pytest.mark.parametrize("expr_str, expected", [
    ("x + 0", "x"),
    ("0 + x", "x"),
    ("x * 1", "x"),
    ("1 * x", "x"),
    ("x**1", "x"),
    ("x**0", "1"),
])
def test_reduce_unit(expr_str, expected):
    expr = sp.sympify(expr_str, evaluate=False)
    node = ExprNode.from_sympy(expr)
    reduced_node = af.reduce_unit(node)
    assert repr(reduced_node) == expected

@pytest.mark.parametrize("expr_str", [
    "x + (-1) * x",
    "x + x * (-1)",
    "(-1) * x + x",
    "x * (-1) + x",
])
def test_cancel(expr_str):
    expr = sp.sympify(expr_str, evaluate=False)
    node = ExprNode.from_sympy(expr)
    cancelled_node = af.cancel(node)
    assert repr(cancelled_node) == "0"
