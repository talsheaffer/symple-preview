from symple.expr.actions import ACTIONS as ACTIONS_BASE, Action
from typing import Tuple
from src.model.tree import ExprNode

def wrap_action(action):
    def wrapper(en):
        return ExprNode.from_expr_node_base(action(en))
    return wrapper

ACTIONS = {
    action_type: Action(action.can_apply, wrap_action(action.apply))
    for action_type, action in ACTIONS_BASE.items()
}

def apply_op_and_count(op_func):
    def wrapper(expr: ExprNode, coord: tuple[int, ...] = ()) -> Tuple[ExprNode, tuple[int, ...], int]:
        initial_count = expr.get_node(coord).node_count()
        new_expr = expr.apply_at_coord(coord, op_func)
        final_count = new_expr.get_node(coord).node_count()
        reduction = initial_count - final_count
        return new_expr, coord, reduction
    return wrapper

ACTIONS = {
    action_type: Action(action.can_apply, apply_op_and_count(action.apply))
    for action_type, action in ACTIONS.items()
}

