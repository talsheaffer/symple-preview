import unittest

import sympy as sp

from src.model.tree import ExprNode
from src.model.environment import Symple
from src.model.model import SympleAgent
from train.aux_policies import random_policy

x, y, z = sp.symbols('x y z')

expressions = [
    (x + y)**3 - (x**3 + y**3 + 3*x*y*(x + y)),
    (x - y)**4 - (x**4 - 4*x**3*y + 6*x**2*y**2 - 4*x*y**3 + y**4),
    (x + y + z)**2 - (x**2 + y**2 + z**2 + 2*x*y + 2*y*z + 2*z*x),
    x**4 - y**4 - (x**2 + y**2)*(x + y)*(x - y),
    (x + y)**5 - (x**5 + 5*x**4*y + 10*x**3*y**2 + 10*x**2*y**3 + 5*x*y**4 + y**5),
    (x**2 + 2*x*y + y**2) - (x + y)**2,
    sp.sqrt(x**2 + y**2) - sp.sqrt((x + y)**2 - 2*x*y),
    sp.sqrt(x) + sp.sqrt(y) - sp.sqrt(x + y + 2*sp.sqrt(x*y)),
    (sp.sqrt(x) + sp.sqrt(y))**2 - (x + y + 2*sp.sqrt(x*y)),
    sp.sqrt(x**2 + y**2 + z**2) - sp.sqrt(x**2 + (sp.sqrt(y**2 + z**2))**2),
    (x + sp.sqrt(x**2 - 1))*(x - sp.sqrt(x**2 - 1)) - 1,
    sp.sqrt(1 + sp.sqrt(1 + sp.sqrt(1 + x))) - sp.sqrt(x + 2),
]

def check_equality(expr1, expr2):
    x, y, z = sp.symbols('x y z')
    diff = sp.expand(expr1 - expr2)
    return all(diff.subs({var: val}).evalf() == 0
               for var in (x, y, z)
               for val in range(-10, 11))

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
        history, final_state = self.agent(initial_expr)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)

    def test_agent_with_complex_expression(self):
        x, y = sp.symbols("x y")
        expr = sp.expand((x**2 - y + 1) ** 4 + (y**2 - x + 1) ** 3)
        history, final_state = self.agent(expr)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)

    def test_agent_with_single_variable_expression(self):
        x = sp.Symbol("x")
        expr = sp.expand(x**10)
        history, final_state = self.agent(expr)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)

    def test_agent_with_no_variable_expression(self):
        expr = sp.expand(42)
        initial_expr = ExprNode.from_sympy(expr)
        history, final_state = self.agent(initial_expr)
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)

    def test_off_policy_forward(self):
        x = sp.Symbol("x")
        expr = sp.expand((x**2 - x + 1) ** 2)
        initial_expr = ExprNode.from_sympy(expr)

        behavior_policy = random_policy

        history, final_state = self.agent(
            initial_expr,
            behavior_policy=behavior_policy,
            min_steps=20,
            max_steps=1000
        )
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history, list)
        self.assertIsInstance(final_state, ExprNode)
        self.assertTrue('behavior_probability' in history[0].keys())
    

    def test_agent_with_random_policy(self):
        for expr in expressions:
            initial_expr = ExprNode.from_sympy(expr)
            history, final_state = self.agent(
                initial_expr,
                behavior_policy=random_policy,
                min_steps=20,
                max_steps=1000
            )
            
            self.assertGreater(len(history), 0)
            self.assertIsInstance(history, list)
            self.assertIsInstance(final_state, ExprNode)
            
            for event in history:
                if event['action_type'] in ['high_level', 'internal']:
                    self.assertAlmostEqual(event['behavior_probability'].item(), event['target_probability'].item())

            final_sympy = final_state.to_sympy()
            self.assertTrue(check_equality(expr, final_sympy),
                            f"Expressions not equal:\nOriginal: {expr}\nFinal: {final_sympy}")

 

    def test_agent_with_temperature_policy(self):
        for expr in expressions:
            initial_expr = ExprNode.from_sympy(expr)
            history, final_state = self.agent(
                initial_expr,
                behavior_policy=('temperature', 5.0),
                min_steps=20,
                max_steps=1000
            )
            
            self.assertGreater(len(history), 0)
            self.assertIsInstance(history, list)
            self.assertIsInstance(final_state, ExprNode)
            
            final_sympy = final_state.to_sympy()
            self.assertTrue(check_equality(expr, final_sympy),
                            f"Expressions not equal:\nOriginal: {expr}\nFinal: {final_sympy}")



if __name__ == "__main__":
     unittest.main()
