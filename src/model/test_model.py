import unittest
import sympy as sp

from src.model.tree import ExprNode
from src.model.environment import Symple
from src.model.model import SympleAgent


class TestSympleAgent(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 8
        self.global_hidden_size = 16
        self.lstm_n_layers = 2
        self.agent = SympleAgent(self.hidden_size, self.global_hidden_size, lstm_n_layers=self.lstm_n_layers)
        self.env = Symple()

    def test_agent_with_simple_expression(self):
        x = sp.Symbol("x")
        expr = sp.expand((x**2 - x + 1) ** 4)
        initial_expr = ExprNode.from_sympy(expr)
        history, final_state = self.agent(initial_expr, self.env)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)

    def test_agent_with_complex_expression(self):
        x, y = sp.symbols("x y")
        expr = sp.expand((x**2 - y + 1) ** 4 + (y**2 - x + 1) ** 3)
        initial_expr = ExprNode.from_sympy(expr)
        history, final_state = self.agent(initial_expr, self.env)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)

    def test_agent_with_single_variable_expression(self):
        x = sp.Symbol("x")
        expr = sp.expand(x**10)
        initial_expr = ExprNode.from_sympy(expr)
        history, final_state = self.agent(initial_expr, self.env)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)

    def test_agent_with_no_variable_expression(self):
        expr = sp.expand(42)
        initial_expr = ExprNode.from_sympy(expr)
        history, final_state = self.agent(initial_expr, self.env)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)

    def test_off_policy_forward(self):
        x = sp.Symbol("x")
        expr = sp.expand((x**2 - x + 1) ** 2)
        initial_expr = ExprNode.from_sympy(expr)

        def behavior_policy(state, validity_mask):
            return validity_mask.view((1, self.env.num_ops))/validity_mask.sum()

        history, final_state = self.agent(initial_expr, self.env, behavior_policy)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)
        self.assertTrue('behavior_probability' in history[0])


if __name__ == "__main__":
    unittest.main()
