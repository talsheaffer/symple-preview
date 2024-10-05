from symple.expr.actions import ACTIONS, ActionType, Action
from symple.expr.expr_node import ExprNode as ExprNodeBase
from typing import Tuple, Callable, Any, Optional
from src.model.tree import ExprNode
from src.model.state import SympleState
import sympy as sp

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

def apply_op_and_count(op: Callable[[ExprNode], ExprNode]) -> Callable[[SympleState], Tuple[SympleState, int]]:
    def wrapper(state: SympleState) -> Tuple[SympleState, int]:
        initial_count = state.node_count()
        state.en = state.en.apply_at_coord(state.coord, op)
        final_count = state.node_count()
        reduction = (initial_count - final_count)# * state.count_symbol()
        return state, reduction
    return wrapper


def wrap_action(
        action: Callable[[ExprNodeBase], ExprNodeBase],
        depth: int = 0
) -> Callable[[SympleState], Tuple[SympleState, int]]:
    
    def subclass_wrapper(expr: ExprNodeBase) -> ExprNode:
        return ExprNode.from_expr_node_base(action(expr))
    
    wrapper = apply_op_and_count(subclass_wrapper)
    
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

# Add sympy functions to the OPS_MAP


sympy_functions = [
    sp.expand,
    sp.factor,
    sp.cancel,
    sp.together,
    # sp.apart,
    # sp.collect,
    sp.simplify,
    # sp.trigsimp,
    # sp.powsimp
    # sp.expand_power_exp,
]

def find_match(node: ExprNode, node_to_match: ExprNode) -> Optional[ExprNode]:
    if node == node_to_match:
        return node
    if node.left is not None:
        left_match = find_match(node.left, node_to_match)
        if left_match is not None:
            return left_match
    if node.right is not None:
        right_match = find_match(node.right, node_to_match)
        if right_match is not None:
            return right_match
    return None

def substitute_matches(old_node: ExprNode, new_node: ExprNode) -> ExprNode:
    node = find_match(old_node, new_node)
    if node is not None:
        return node
    else:
        new_node.left = substitute_matches(old_node, new_node.left) if new_node.left is not None else None
        new_node.right = substitute_matches(old_node, new_node.right) if new_node.right is not None else None
        return new_node


def wrap_sympy_function(sympy_func: Callable[[Any], Any]) -> Callable[[SympleState], Tuple[SympleState, int]]:
    def wrapper(state: SympleState) -> Tuple[SympleState, int]:
        initial_count = state.node_count()
        old_node = state.current_node
        sympy_expr = state.to_sympy(old_node)
        result = sympy_func(sympy_expr)
        new_node = state.Expr_Node_from_sympy(result)
        new_node = substitute_matches(old_node, new_node)
        state.substitute_current_node(new_node)
        final_count = state.node_count()
        reduction = (initial_count - final_count) #* state.count_symbol()
        return state, reduction
    return wrapper

def always_applicable(state: SympleState) -> bool:
    return True

for func in sympy_functions:
    wrapped_func = wrap_sympy_function(func)
    OPS_MAP.append(Action(always_applicable, wrapped_func))



# Add SympleState actions to the OPS_MAP

def can_declare_new_symbol(state: SympleState) -> bool:
    return state.can_declare_symbol()

def declare_new_symbol(state: SympleState) -> Tuple[SympleState, int]:
    state.declare_new_symbol()
    return state, 0

def can_switch_to_sub_state(state: SympleState) -> bool:
    return state.can_switch_to_sub_state()

def switch_to_sub_state(state: SympleState) -> Tuple[SympleState, int]:
    state.switch_to_sub_state()
    return state, 0

def can_revert_to_primary_state(state: SympleState) -> bool:
    return state.can_revert_to_primary_state()

def revert_to_primary_state(state: SympleState) -> Tuple[SympleState, int]:
    state.revert_to_primary_state()
    return state, 0

def can_evaluate_symbol(state: SympleState) -> bool:
    return state.can_evaluate_symbol()

def evaluate_symbol(state: SympleState) -> Tuple[SympleState, int]:
    state.evaluate_symbol()
    return state, 0

OPS_MAP.append(Action(can_declare_new_symbol, declare_new_symbol))
OPS_MAP.append(Action(can_switch_to_sub_state, switch_to_sub_state))
OPS_MAP.append(Action(can_revert_to_primary_state, revert_to_primary_state))
OPS_MAP.append(Action(can_evaluate_symbol, evaluate_symbol))


