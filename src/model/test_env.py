import unittest

from src.model.environment import(
    Symple, 
    # OPS_MAP, 
    NUM_OPS, 
    OP_FINISH, 
    OP_MOVE_UP, 
    OP_MOVE_LEFT, 
    OP_MOVE_RIGHT, 
    # OP_DISTRIBUTE_B, 
    # OP_UNDISTRIBUTE_B, 
    OP_REDUCE_UNIT
)
from src.model.tree import(
    ExprNode, 
    ADD_TYPE, 
    # MUL_TYPE, 
    X_TYPE, 
    Y_TYPE, 
    # Z_TYPE
)
import sympy as sp

class TestSympleEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = Symple()
        
        # Define test cases globally for all tests
        self.test_cases = [
            (ExprNode.from_sympy(sp.Symbol('x')), ()),  # Single variable
            (ExprNode.from_sympy(sp.sympify('x + y')), ()),  # Simple addition
            (ExprNode.from_sympy(sp.sympify('x * y')), ()),  # Simple multiplication
            (ExprNode.from_sympy(sp.sympify('x + y')), (0,)),  # Left child of addition
            (ExprNode.from_sympy(sp.sympify('(x + y) * z')), ()),  # Complex expression
        ]

    def test_validity_mask(self):
        # Test various expressions with validity mask
        for expr, coord in self.test_cases:
            with self.subTest(expr=expr, coord=coord):
                validity_mask = self.env.get_validity_mask(expr, coord)
                self.assertEqual(len(validity_mask), NUM_OPS)
                
                for op in range(NUM_OPS):
                    if validity_mask[op]:
                        try:
                            self.env.step(expr, coord, op)
                        except Exception as e:
                            self.fail(f"Valid operation {op} raised an exception: {e}")
                    else:
                        with self.assertRaises(Exception):
                            self.env.step(expr, coord, op)

        # Test specific cases
        expr = ExprNode(ADD_TYPE, a=ExprNode(X_TYPE), b=ExprNode(Y_TYPE))
        validity_mask = self.env.get_validity_mask(expr)
        
        self.assertTrue(validity_mask[OP_FINISH])
        self.assertTrue(validity_mask[OP_MOVE_LEFT])
        self.assertTrue(validity_mask[OP_MOVE_RIGHT])
        self.assertFalse(validity_mask[OP_MOVE_UP])  # Can't move up from root
        
        # Move to left child and check validity
        validity_mask = self.env.get_validity_mask(expr, (0,))
        self.assertTrue(validity_mask[OP_MOVE_UP])
        self.assertFalse(validity_mask[OP_MOVE_LEFT])  # Can't move left from leaf
        self.assertFalse(validity_mask[OP_MOVE_RIGHT])  # Can't move right from leaf
        
    def test_step_node_count(self):
        for expr, coord in self.test_cases:
            initial_count = expr.node_count()
            validity_mask = self.env.get_validity_mask(expr, coord)

            # Test all operations
            for op in range(NUM_OPS):
                if validity_mask[op]:
                    new_expr, _, _, node_count_reduction, _ = self.env.step(expr, coord, op)
                    final_count = new_expr.node_count()
                    expected_reduction = initial_count - final_count
                    self.assertEqual(node_count_reduction, expected_reduction,
                                     f"Operation {op} failed: expected reduction {expected_reduction}, got {node_count_reduction}")

    def test_step_rewards(self):
        for expr, coord in self.test_cases:
            validity_mask = self.env.get_validity_mask(expr, coord)
            
            # Test time penalty
            if validity_mask[OP_MOVE_LEFT]:
                _, _, reward, _, _ = self.env.step(expr, coord, OP_MOVE_LEFT)
                self.assertAlmostEqual(reward, self.env.time_penalty)

            # Test node count reduction reward
            env_with_high_importance = Symple(node_count_importance_factor=10.0)
            validity_mask = env_with_high_importance.get_validity_mask(expr, coord)
            if validity_mask[OP_REDUCE_UNIT]:
                _, _, reward, _, _ = env_with_high_importance.step(expr, coord, OP_REDUCE_UNIT)
                self.assertGreater(reward, 0)  # Assuming REDUCE_UNIT reduces node count

    def test_done_conditions(self):
        for expr, coord in self.test_cases:
            # Test FINISH operation
            validity_mask = self.env.get_validity_mask(expr, coord)
            if validity_mask[OP_FINISH]:
                _, _, _, _, done = self.env.step(expr, coord, OP_FINISH)
                self.assertTrue(done)

            # Test max steps
            env_with_max_steps = Symple(max_steps=1)
            validity_mask = env_with_max_steps.get_validity_mask(expr, coord)
            for op in range(NUM_OPS):
                if validity_mask[op] and op != OP_FINISH:
                    _, _, _, _, done = env_with_max_steps.step(expr, coord, op)
                    self.assertTrue(done)
                    break

if __name__ == '__main__':
    unittest.main()
