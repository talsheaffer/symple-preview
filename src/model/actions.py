from symple.expr.actions import ACTIONS, ActionType
from symple.expr.expr_node import ExprNode as ExprNodeBase
from typing import Tuple, Callable
from src.model.tree import ExprNode

depths = {key: 1 for key in ACTIONS.keys()}
depths[ActionType.ASSOCIATE_LEFT] = 2
depths[ActionType.ASSOCIATE_RIGHT] = 2
depths[ActionType.DISTRIBUTE_LEFT] = 2
depths[ActionType.DISTRIBUTE_RIGHT] = 2
depths[ActionType.FACTOR_RIGHT] = 2
depths[ActionType.FACTOR_LEFT] = 2
depths[ActionType.REDUCE_DOUBLE_INV] = 0
depths[ActionType.REDUCE_GROUP_UNIT] = 0
depths[ActionType.MULTIPLY_ONE] = 2
depths[ActionType.ADD_ZERO] = 2




def wrap_action(
        action: Callable[[ExprNodeBase], ExprNodeBase],
        depth: int = 0
) -> Callable[[ExprNodeBase, tuple[int, ...]], Tuple[ExprNode, tuple[int, ...], int]]:
    
    def subclass_wrapper(expr: ExprNodeBase) -> ExprNode:
        return ExprNode.from_expr_node_base(action(expr))
    
    def wrapper(expr: ExprNodeBase, coord: tuple[int, ...] = ()) -> Tuple[ExprNode, tuple[int, ...], int]:
        initial_count = expr.get_node(coord).node_count()
        # nodes = expr.get_node(coord).topological_sort(depth)
        # tensors = [(n.hidden, n.cell) for n in nodes]
        new_expr = expr.apply_at_coord(coord, subclass_wrapper)
        final_count = new_expr.get_node(coord).node_count()
        reduction = initial_count - final_count
        # nodes = new_expr.get_node(coord).topological_sort(depth)
        # for i in range(len(tensors)):
        #     if i < len(nodes):
        #         nodes[-i-1].hidden = tensors[-i-1][0]
        #         nodes[-i-1].cell = tensors[-i-1][1]
        return new_expr, coord, reduction
    return wrapper

for action_type, action in ACTIONS.items():
    action.apply = wrap_action(action.apply, depth = depths[action_type])
