from symple.expr.actions import ACTIONS, ActionType, Action
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
def wrap_can_apply(can_apply: Callable[[ExprNodeBase], bool]) -> Callable[[SympleState], bool]:
    def wrapper(state: SympleState) -> bool:
        return can_apply(state.en.get_node(state.coord))
    return wrapper

for action_type, action in ACTIONS.items():
    action.apply = wrap_action(action.apply, depth = depths[action_type])
    action.can_apply = wrap_can_apply(action.can_apply)

OPS_MAP = list(ACTIONS.values())


def can_move_up(state: SympleState) -> bool:
    return len(state.coord) > 0

def can_move_left(state: SympleState) -> bool:
    return state.en.get_node(state.coord).left is not None

def can_move_right(state: SympleState) -> bool:
    return state.en.get_node(state.coord).right is not None

def move_up(state: SympleState) -> Tuple[SympleState, int]:
    state.coord = state.coord[:-1]
    return state, 0

def move_left(state: SympleState) -> Tuple[SympleState, int]:
    state.coord = state.coord + (0,)
    return state, 0

def move_right(state: SympleState) -> Tuple[SympleState, int]:
    state.coord = state.coord + (1,)
    return state, 0

# Add new actions to the OPS_MAP
OPS_MAP.append(Action(can_move_up, move_up))
OPS_MAP.append(Action(can_move_left, move_left))
OPS_MAP.append(Action(can_move_right, move_right))