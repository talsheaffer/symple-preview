import unittest

import sympy as sp
import torch

from src.model.tree import (ADD_TYPE, INT_NE_TYPE, INT_PO_TYPE, MUL_TYPE,
                            POW_TYPE, X_TYPE, Y_TYPE, Z_TYPE, ExprNode)


class TestExprNode(unittest.TestCase):

    def test_addition(self):
        expr = sp.sympify('x + y', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(node.type, ADD_TYPE)
        self.assertEqual(node.arg, 0)
        self.assertEqual(node.a.type, X_TYPE)
        self.assertEqual(node.b.type, Y_TYPE)

    def test_multiplication(self):
        expr = sp.sympify('x * y', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(node.type, MUL_TYPE)
        self.assertEqual(node.arg, 0)
        self.assertEqual(node.a.type, X_TYPE)
        self.assertEqual(node.b.type, Y_TYPE)

    def test_power(self):
        expr = sp.sympify('x**2', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(node.type, POW_TYPE)
        self.assertEqual(node.arg, 0)
        self.assertEqual(node.a.type, X_TYPE)
        self.assertEqual(node.b.type, INT_PO_TYPE)
        self.assertEqual(node.b.arg, 2)

    def test_integer(self):
        expr = sp.sympify('-3', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(node.type, INT_NE_TYPE)
        self.assertEqual(node.arg, 3)

        expr = sp.sympify('3', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(node.type, INT_PO_TYPE)
        self.assertEqual(node.arg, 3)

    def test_symbol(self):
        expr = sp.sympify('x', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(node.type, X_TYPE)
        self.assertEqual(node.arg, 0)

    def test_to_tensor(self):
        expr = sp.sympify('x + y', evaluate=False)
        node = ExprNode.from_sympy(expr)
        tensor = node.to_tensor()
        self.assertTrue(torch.equal(tensor, torch.tensor([ADD_TYPE, 0], dtype=tensor.dtype)))

    def test_topological_sort(self):
        expr = sp.sympify('x + y * z', evaluate=False)
        node = ExprNode.from_sympy(expr)
        sorted_nodes = node.topological_sort()
        self.assertEqual(len(sorted_nodes), 5)
        self.assertEqual(sorted_nodes[0].type, X_TYPE)
        self.assertEqual(sorted_nodes[1].type, Y_TYPE)
        self.assertEqual(sorted_nodes[2].type, Z_TYPE)
        self.assertEqual(sorted_nodes[3].type, MUL_TYPE)
        self.assertEqual(sorted_nodes[4].type, ADD_TYPE)

    def test_addition_with_parentheses(self):
        expr = sp.sympify('(x + y) * z', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(repr(node), '((x + y) * z)')

    def test_multiplication_with_parentheses(self):
        expr = sp.sympify('x * (y * z)', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(repr(node), '(x * (y * z))')

    def test_mixed_operations_with_parentheses(self):
        expr = sp.sympify('(x + y) * (z + 2)', evaluate=False)
        node = ExprNode.from_sympy(expr)
        self.assertEqual(repr(node), '((x + y) * (z + 2))')

    def test_commute(self):
        expr = sp.sympify('x + y', evaluate=False)
        node = ExprNode.from_sympy(expr)
        commuted_node = node.commute()
        self.assertEqual(repr(commuted_node), '(y + x)')

    def test_associate_b(self):
        expr = sp.sympify('x + (y + z)', evaluate=False)
        node = ExprNode.from_sympy(expr)
        associated_node = node.associate_b()
        self.assertEqual(repr(associated_node), '((x + y) + z)')

    def test_distribute_b(self):
        expr = sp.sympify('x * (y + z)', evaluate=False)
        node = ExprNode.from_sympy(expr)
        distributed_node = node.distribute_b()
        self.assertEqual(repr(distributed_node), '((x * y) + (x * z))')

    def test_undistribute_b(self):
        expr = sp.sympify('(x * y) + (x * z)', evaluate=False)
        node = ExprNode.from_sympy(expr)
        undistributed_node = node.undistribute_b()
        self.assertEqual(repr(undistributed_node), '(x * (y + z))')

    def test_reduce_unit(self):
        expr = sp.sympify('x + 0', evaluate=False)
        node = ExprNode.from_sympy(expr)
        reduced_node = node.reduce_unit()
        self.assertEqual(repr(reduced_node), 'x')
        
        expr = sp.sympify('0 + x', evaluate=False)
        node = ExprNode.from_sympy(expr)
        reduced_node = node.reduce_unit()
        self.assertEqual(repr(reduced_node), 'x')

        expr = sp.sympify('x * 1', evaluate=False)
        node = ExprNode.from_sympy(expr)
        reduced_node = node.reduce_unit()
        self.assertEqual(repr(reduced_node), 'x')
        
        expr = sp.sympify('1 * x', evaluate=False)
        node = ExprNode.from_sympy(expr)
        reduced_node = node.reduce_unit()
        self.assertEqual(repr(reduced_node), 'x')

        expr = sp.sympify('x**1', evaluate=False)
        node = ExprNode.from_sympy(expr)
        reduced_node = node.reduce_unit()
        self.assertEqual(repr(reduced_node), 'x')

        expr = sp.sympify('x**0', evaluate=False)
        node = ExprNode.from_sympy(expr)
        reduced_node = node.reduce_unit()
        self.assertEqual(repr(reduced_node), '1')

    def test_cancel(self):
        def _f(expr_str):
            expr = sp.sympify(expr_str, evaluate=False)
            node = ExprNode.from_sympy(expr)
            cancelled_node = node.cancel()
            self.assertEqual(repr(cancelled_node), '0')

        _f('x + (-1) * x')
        _f('x + x * (-1)')
        _f('(-1) * x + x')
        _f('x * (-1) + x')


if __name__ == '__main__':
    unittest.main()