from typing import Tuple#, Optional

import torch

from src.model.tree import ExprNode
from src.model.actions import ACTIONS as OPS_MAP

import sympy as sp

TIME_PENALTY = -0.0002
NODE_COUNT_IMPORTANCE_FACTOR = 1.0
COMPUTE_PENALTY_COEFFICIENT = 0


NUM_OPS = len(OPS_MAP)





# Consider using Open-Ai Gym?
class Symple:
    """
    An RL environment with which the agent should interact. To enrich the set
    of operations we must update the variable OPS and the method Symple.get_validity_mask.
    The operations should return (state: ExprNode, reward: float, done: bool).
    """

    def __init__(
        self,
        time_penalty: float = TIME_PENALTY,
        node_count_importance_factor: float = NODE_COUNT_IMPORTANCE_FACTOR,
        compute_penalty_coefficient: float = COMPUTE_PENALTY_COEFFICIENT,
    ):
        assert time_penalty <= 0, "Time penalty must be non-positive"
        self.time_penalty = time_penalty
        self.node_count_importance_factor = node_count_importance_factor
        self.compute_penalty_coefficient = compute_penalty_coefficient
        self.num_ops = NUM_OPS
    def step(self, expr: ExprNode, current_coord: tuple[int, ...], action: int) -> Tuple[ExprNode, tuple[int, ...], float, bool]:
        
        new_expr, new_coord, node_count_reduction = expr.apply_at_coord(current_coord, OPS_MAP[action].apply)
        
        reward = self.time_penalty + self.node_count_importance_factor * node_count_reduction
        

        return new_expr, new_coord, reward, node_count_reduction

    def get_validity_mask(self, expr: ExprNode, coord: tuple[int, ...] = ()) -> torch.Tensor:
        current_node = expr.get_node(coord)        
        validity_mask = torch.ones(NUM_OPS, dtype=int)

        for op_type, op in OPS_MAP.items():
            validity_mask[op_type] = op.can_apply(current_node)
        
        return validity_mask

    @staticmethod
    def from_sympy(expr: sp.Expr, **kwargs) -> Tuple[ExprNode, tuple[int, ...], "Symple"]:
        initial_expr = ExprNode.from_sympy(expr)
        initial_coord = ()
        env = Symple(**kwargs)
        return initial_expr, initial_coord, env