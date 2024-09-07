import unittest

import sympy as sp

from model import ExprNode, Symple, SympleAgent


class TestSympleAgent(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 8
        self.embedding_size = 8
        self.agent = SympleAgent(self.hidden_size, self.embedding_size)

    def test_agent_with_simple_expression(self):
        x = sp.Symbol("x")
        expr = sp.expand((x**2 - x + 1) ** 4)
        en = ExprNode.from_sympy(expr)
        env = Symple(en)
        rewards = self.agent(env)
        self.assertGreater(len(rewards), 0)
        self.assertIsInstance(rewards, list)

    def test_agent_with_complex_expression(self):
        x, y = sp.symbols("x y")
        expr = sp.expand((x**2 - y + 1) ** 4 + (y**2 - x + 1) ** 3)
        en = ExprNode.from_sympy(expr)
        env = Symple(en)
        rewards = self.agent(env)
        self.assertGreater(len(rewards), 0)
        self.assertIsInstance(rewards, list)

    def test_agent_with_single_variable_expression(self):
        x = sp.Symbol("x")
        expr = sp.expand(x**10)
        en = ExprNode.from_sympy(expr)
        env = Symple(en)
        rewards = self.agent(env)
        self.assertGreater(len(rewards), 0)
        self.assertIsInstance(rewards, list)

    def test_agent_with_no_variable_expression(self):
        expr = sp.expand(42)
        en = ExprNode.from_sympy(expr)
        env = Symple(en)
        rewards = self.agent(env)
        self.assertGreater(len(rewards), 0)
        self.assertIsInstance(rewards, list)


if __name__ == "__main__":
    unittest.main()
