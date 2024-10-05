from typing import Tuple

import torch

from src.model.actions import OPS_MAP
from src.model.state import SympleState


TIME_PENALTY = -0.0002
NODE_COUNT_IMPORTANCE_FACTOR = 1.0
COMPUTE_PENALTY_COEFFICIENT = 0


NUM_OPS = len(OPS_MAP)






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
        state, node_count_reduction = OPS_MAP[action].apply(state)

        assert state.nc == state.node_count() + node_count_reduction, f"Node count reduction does not match: {state.nc} != {state.node_count()} + {node_count_reduction}"
        
        reward = self.time_penalty + self.node_count_importance_factor * node_count_reduction
        
        return state, reward, node_count_reduction

    def get_validity_mask(self, state: SympleState) -> torch.Tensor:    
        validity_mask = torch.ones(NUM_OPS, dtype=int)

        for op_type, op in enumerate(OPS_MAP):
            validity_mask[op_type] = op.can_apply(state)
        
        return validity_mask
