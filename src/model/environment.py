from typing import Tuple#, Optional

import torch

from src.model.tree import ExprNode, INF_TYPE
from src.utils import iota

import sympy as sp

def apply_op_for_no_change(op_func):
    def wrapper(expr: ExprNode, coord: tuple[int, ...]) -> Tuple[ExprNode, tuple[int, ...], int]:
        new_expr = expr.apply_at_coord(coord, op_func)
        return new_expr, coord, 0
    return wrapper

def apply_op_and_count(op_func):
    def wrapper(expr: ExprNode, coord: tuple[int, ...]) -> Tuple[ExprNode, tuple[int, ...], int]:
        initial_count = expr.get_node(coord).node_count()
        new_expr = expr.apply_at_coord(coord, op_func)
        final_count = new_expr.get_node(coord).node_count()
        reduction = initial_count - final_count
        return new_expr, coord, reduction
    return wrapper


it = iota()
OPS_MAP = []

OP_FINISH = next(it)
@apply_op_for_no_change
def op_finish(expr: ExprNode) -> ExprNode:
    return expr
OPS_MAP.append(op_finish)

OP_MOVE_UP = next(it)
def op_move_up(expr: ExprNode, coord: tuple[int, ...]) -> Tuple[ExprNode, tuple[int, ...], int]:
    new_coord = coord[:-1] if coord else coord
    return expr, new_coord, 0
OPS_MAP.append(op_move_up)

OP_MOVE_LEFT = next(it)    
def op_move_left(expr: ExprNode, coord: tuple[int, ...]) -> Tuple[ExprNode, tuple[int, ...], int]:
    new_coord = coord + (0,)
    return expr, new_coord, 0
OPS_MAP.append(op_move_left)

OP_MOVE_RIGHT = next(it)
def op_move_right(expr: ExprNode, coord: tuple[int, ...]) -> Tuple[ExprNode, tuple[int, ...], int]:
    new_coord = coord + (1,)
    return expr, new_coord, 0
OPS_MAP.append(op_move_right)

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
    An RL environment with which the agent should interact. To enrich the set
    of operations we must update the variable OPS and the method Symple.get_validity_mask.
    The operations should return (state: ExprNode, reward: float, done: bool).
    """

    def __init__(
        self,
        time_penalty: float = -0.02,
        node_count_importance_factor: float = 1.0,
        min_steps: int = 0,
        max_steps: int = 1000,
    ):
        self.time_penalty = time_penalty
        self.node_count_importance_factor = node_count_importance_factor
        self.min_steps = min_steps

    def step(self, expr: ExprNode, current_coord: tuple[int, ...], action: int) -> Tuple[ExprNode, tuple[int, ...], float, bool]:
        reward = self.time_penalty if self.min_steps == 0 else 0.
        
        new_expr, new_coord, node_count_reduction = OPS_MAP[action](expr, current_coord)
        
        reward += self.node_count_importance_factor * node_count_reduction
        
        if self.max_steps > 0:
            self.max_steps -= 1
        if self.min_steps > 0:
            self.min_steps -= 1
        
        
        done = action == OP_FINISH or self.max_steps == 0
        

        return new_expr, new_coord, reward, done

    def get_validity_mask(self, expr: ExprNode, coord: tuple[int, ...] = ()) -> torch.Tensor:
        current_node = expr.get_node(coord)
        # If we are at the infinity node, we must finish.
        if current_node.type == INF_TYPE:
            validity_mask = torch.zeros(NUM_OPS, dtype=int)
            validity_mask[OP_FINISH] = 1
            return validity_mask
        
        validity_mask = torch.ones(NUM_OPS, dtype=int)

        validity_mask[OP_MOVE_UP] = bool(current_node.p)
        validity_mask[OP_MOVE_LEFT] = bool(current_node.a)
        validity_mask[OP_MOVE_RIGHT] = bool(current_node.b)
        validity_mask[OP_COMMUTE] = current_node.can_commute()
        validity_mask[OP_ASSOCIATE_B] = current_node.can_associate_b()
        validity_mask[OP_DISTRIBUTE_B] = current_node.can_distribute_b()
        validity_mask[OP_UNDISTRIBUTE_B] = current_node.can_undistribute_b()
        validity_mask[OP_REDUCE_UNIT] = current_node.can_reduce_unit()
        validity_mask[OP_CANCEL] = current_node.can_cancel()
        validity_mask[OP_FINISH] = self.min_steps <= 0
        validity_mask[OP_FINISH] = validity_mask[OP_FINISH] or not any(validity_mask) # If we can't do anything, we must finish.
        
        return validity_mask

    @staticmethod
    def from_sympy(expr: sp.Expr, **kwargs) -> Tuple[ExprNode, tuple[int, ...], "Symple"]:
        initial_expr = ExprNode.from_sympy(expr)
        initial_coord = ()
        env = Symple(**kwargs)
        return initial_expr, initial_coord, env