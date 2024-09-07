from typing import Tuple

import torch

from src.model.tree import ExprNode
from src.utils import iota

it = iota()
OPS_MAP = list()

OP_FINISH = next(it)
def op_finish(expr: ExprNode) -> ExprNode:
    return expr
OPS_MAP.append(op_finish)

OP_MOVE_UP = next(it)
def op_move_up(expr: ExprNode) -> ExprNode:
    return expr.p
OPS_MAP.append(op_move_up)

OP_MOVE_LEFT = next(it)    
def op_move_left(expr: ExprNode) -> ExprNode:
    return expr.a
OPS_MAP.append(op_move_left)

OP_MOVE_RIGHT = next(it)
def op_move_right(expr: ExprNode) -> ExprNode:
    return expr.b
OPS_MAP.append(op_move_right)

OP_PASS = next(it)
def op_pass(expr: ExprNode) -> ExprNode:
    return expr
OPS_MAP.append(op_pass)

OP_COMMUTE = next(it)
def op_commute(expr: ExprNode) -> ExprNode:
    return expr.commute()
OPS_MAP.append(op_commute)

OP_ASSOCIATE_B = next(it)
def op_associate_b(expr: ExprNode) -> ExprNode:
    return expr.associate_b()
OPS_MAP.append(op_associate_b)

OP_DISTRIBUTE_B = next(it)
def op_distribute_b(expr: ExprNode) -> ExprNode:
    return expr.distribute_b()
OPS_MAP.append(op_distribute_b)

OP_UNDISTRIBUTE_B = next(it)
def op_undistribute_b(expr: ExprNode) -> ExprNode:
    return expr.undistribute_b()
OPS_MAP.append(op_undistribute_b)

OP_REDUCE_UNIT = next(it)
def op_reduce_unit(expr: ExprNode) -> ExprNode:
    return expr.reduce_unit()
OPS_MAP.append(op_reduce_unit)

OP_CANCEL = next(it)
def op_cancel(expr: ExprNode) -> ExprNode:
    return expr.cancel()
OPS_MAP.append(op_cancel)

NUM_OPS = next(it)


# Consider using Open-Ai Gym?
class Symple:
    """
    An RL environment with which the agent should interact. To enrich the s
    et of operations we must update the variable OPS and the method Symple.update_validity_mask. The operations should return (state : the new ExprNode, reward: float, don
    e : bool).
    """

    def __init__(
        self,
        expr: "ExprNode",
        time_penalty: float = -0.02,
        node_count_importance_factor: float = 1.0,
    ):
        self.expr = expr
        self.state = expr
        self.validity_mask = torch.ones(NUM_OPS, dtype=int)
        self.update_validity_mask()
        self.time_penalty = time_penalty
        self.node_count_importance_factor = node_count_importance_factor

    def step(self, action: int) -> Tuple[float, bool]:
        reward = self.time_penalty
        
        node_count_reduction = self.state.node_count()
        self.state = OPS_MAP[action](self.state)
        node_count_reduction -= self.state.node_count()
        
        self.update_validity_mask()
        reward += self.node_count_importance_factor * node_count_reduction
        return reward, action == OP_FINISH

    def update_validity_mask(self) -> None:
        self.validity_mask[OP_MOVE_UP] = bool(self.state.p)
        self.validity_mask[OP_MOVE_LEFT] = bool(self.state.a)
        self.validity_mask[OP_MOVE_RIGHT] = bool(self.state.b)
        self.validity_mask[OP_COMMUTE] = self.state.can_commute()
        self.validity_mask[OP_ASSOCIATE_B] = self.state.can_associate_b()
        self.validity_mask[OP_DISTRIBUTE_B] = self.state.can_distribute_b()
        self.validity_mask[OP_UNDISTRIBUTE_B] = self.state.can_undistribute_b()
        self.validity_mask[OP_REDUCE_UNIT] = self.state.can_reduce_unit()
        self.validity_mask[OP_CANCEL] = self.state.can_cancel()
        self.validity_mask[OP_FINISH] = True
