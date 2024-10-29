import sympy as sp

from symple.expr import ExprNode, ExprNodeType


def test_addition():
    expr = sp.sympify("x + y", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert node.type == ExprNodeType.ADD_TYPE
    assert node.arg == 0
    assert node.a.type == ExprNodeType.X_TYPE
    assert node.b.type == ExprNodeType.Y_TYPE

def test_multiplication():
    expr = sp.sympify("x * y", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert node.type == ExprNodeType.MUL_TYPE
    assert node.arg == 0
    assert node.a.type == ExprNodeType.X_TYPE
    assert node.b.type == ExprNodeType.Y_TYPE

def test_power():
    expr = sp.sympify("x**2", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert node.type == ExprNodeType.POW_TYPE
    assert node.arg == 0
    assert node.a.type == ExprNodeType.X_TYPE
    assert node.b.type == ExprNodeType.INT_PO_TYPE
    assert node.b.arg == 2

def test_integer():
    expr = sp.sympify("-3", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert node.type == ExprNodeType.INT_NE_TYPE
    assert node.arg == 3

    expr = sp.sympify("3", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert node.type == ExprNodeType.INT_PO_TYPE
    assert node.arg == 3

def test_symbol():
    expr = sp.sympify("x", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert node.type == ExprNodeType.X_TYPE
    assert node.arg == 0

def test_topological_sort():
    expr = sp.sympify("x + y * z", evaluate=False)
    node = ExprNode.from_sympy(expr)
    sorted_nodes = node.topological_sort()
    assert len(sorted_nodes) == 5
    assert sorted_nodes[0].type == ExprNodeType.X_TYPE
    assert sorted_nodes[1].type == ExprNodeType.Y_TYPE
    assert sorted_nodes[2].type == ExprNodeType.Z_TYPE
    assert sorted_nodes[3].type == ExprNodeType.MUL_TYPE
    assert sorted_nodes[4].type == ExprNodeType.ADD_TYPE

def test_addition_with_parentheses():
    expr = sp.sympify("(x + y) * z", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert repr(node) == "((x + y) * z)"

def test_multiplication_with_parentheses():
    expr = sp.sympify("x * (y * z)", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert repr(node) == "(x * (y * z))"

def test_mixed_operations_with_parentheses():
    expr = sp.sympify("(x + y) * (z + 2)", evaluate=False)
    node = ExprNode.from_sympy(expr)
    assert repr(node) == "((x + y) * (z + 2))"

def test_clone():
    expr = sp.sympify("x + y * z", evaluate=False)
    original_node = ExprNode.from_sympy(expr)
    cloned_node = original_node.clone()

    # Check that the cloned node is equal but not the same object
    assert original_node == cloned_node
    assert original_node is not cloned_node

    # Check that the structure is preserved
    assert original_node.type == cloned_node.type
    assert original_node.arg == cloned_node.arg
    assert original_node.a.type == cloned_node.a.type
    assert original_node.b.type == cloned_node.b.type

    # Check that modifying the clone doesn't affect the original
    cloned_node.b.a.type = ExprNodeType.X_TYPE
    assert original_node.b.a.type == ExprNodeType.Y_TYPE
    assert cloned_node.b.a.type == ExprNodeType.X_TYPE

    # Check that deep structure is also cloned
    assert original_node.b.b is not cloned_node.b.b