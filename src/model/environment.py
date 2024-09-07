from typing import Tuple

import torch

from src.model.tree import ExprNode
from src.utils import iota

import sympy as sp


def apply_op_and_count(op_func):
    def wrapper(expr: ExprNode) -> Tuple[ExprNode, int]:
        initial_count = expr.node_count()
        result = op_func(expr)
        final_count = result.node_count()
        reduction = initial_count - final_count
        return result, reduction
    return wrapper


it = iota()
OPS_MAP = []

OP_FINISH = next(it)
def op_finish(expr: ExprNode) -> Tuple[ExprNode, int]:
    return expr, 0
OPS_MAP.append(op_finish)

OP_MOVE_UP = next(it)
def op_move_up(expr: ExprNode) -> Tuple[ExprNode, int]:
    return expr.p, 0
OPS_MAP.append(op_move_up)

OP_MOVE_LEFT = next(it)    
def op_move_left(expr: ExprNode) -> Tuple[ExprNode, int]:
    return expr.a, 0
OPS_MAP.append(op_move_left)

OP_MOVE_RIGHT = next(it)
def op_move_right(expr: ExprNode) -> Tuple[ExprNode, int]:
    return expr.b, 0
OPS_MAP.append(op_move_right)

OP_PASS = next(it)
def op_pass(expr: ExprNode) -> Tuple[ExprNode, int]:
    return expr, 0
OPS_MAP.append(op_pass)

OP_COMMUTE = next(it)
@apply_op_and_count
def op_commute(expr: ExprNode) -> ExprNode:
    return expr.commute()
OPS_MAP.append(op_commute)

OP_ASSOCIATE_B = next(it)
@apply_op_and_count
def op_associate_b(expr: ExprNode) -> ExprNode:
    return expr.associate_b()
OPS_MAP.append(op_associate_b)

OP_DISTRIBUTE_B = next(it)
@apply_op_and_count
def op_distribute_b(expr: ExprNode) -> ExprNode:
    return expr.distribute_b()
OPS_MAP.append(op_distribute_b)

OP_UNDISTRIBUTE_B = next(it)
@apply_op_and_count
def op_undistribute_b(expr: ExprNode) -> ExprNode:
    return expr.undistribute_b()
OPS_MAP.append(op_undistribute_b)

OP_REDUCE_UNIT = next(it)
@apply_op_and_count
def op_reduce_unit(expr: ExprNode) -> ExprNode:
    return expr.reduce_unit()
OPS_MAP.append(op_reduce_unit)

OP_CANCEL = next(it)
@apply_op_and_count
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
        
        self.state, node_count_reduction = OPS_MAP[action](self.state)
        
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

    @staticmethod
    def from_sympy(expr: sp.Expr) -> "Symple":
        return Symple(ExprNode.from_sympy(expr))