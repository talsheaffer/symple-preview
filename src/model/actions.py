from symple.expr.actions import ACTIONS, ActionType
from symple.expr.expr_node import ExprNode as ExprNodeBase
from typing import Tuple, Callable
from src.model.tree import ExprNode
from src.model.state import SympleState

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
) -> Callable[[SympleState], Tuple[SympleState, int]]:
    
    def subclass_wrapper(expr: ExprNodeBase) -> ExprNode:
        return ExprNode.from_expr_node_base(action(expr))
    
    def wrapper(state: SympleState) -> Tuple[SympleState, int]:
        initial_count = (
            state.en.
            # get_node(state.coord).
            node_count()
        )
        state.en = state.en.apply_at_coord(state.coord, subclass_wrapper)
        final_count = (
            state.en.
            # get_node(state.coord).
            node_count()
        )
        reduction = initial_count - final_count
        
        return state, reduction
    
    return wrapper

for action_type, action in ACTIONS.items():
    action.apply = wrap_action(action.apply, depth = depths[action_type])
