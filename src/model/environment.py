from typing import Tuple#, Optional

import torch

from src.model.tree import ExprNode
from src.model.actions import ACTIONS as OPS_MAP

from dataclasses import dataclass


TIME_PENALTY = -0.0002
NODE_COUNT_IMPORTANCE_FACTOR = 1.0
COMPUTE_PENALTY_COEFFICIENT = 0


NUM_OPS = len(OPS_MAP)




@dataclass
class SympleState:
    en: ExprNode
    coord: tuple[int, ...]
    h_glob: torch.Tensor
    c_glob: torch.Tensor

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

    def step(self, state: SympleState, action: int) -> Tuple[SympleState, float, bool]:
        state.en, state.coord, node_count_reduction = OPS_MAP[action].apply(state.en, state.coord)
        
        reward = self.time_penalty + self.node_count_importance_factor * node_count_reduction
        
        return state, reward, node_count_reduction

    def get_validity_mask(self, state: SympleState) -> torch.Tensor:
        current_node = state.en.get_node(state.coord)        
        validity_mask = torch.ones(NUM_OPS, dtype=int)

        for op_type, op in OPS_MAP.items():
            validity_mask[op_type] = op.can_apply(current_node)
        
        return validity_mask
