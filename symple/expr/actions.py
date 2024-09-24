from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable

from symple.expr.actions_group import ActionGroup
from symple.expr.actions_ints import (
    can_eval_ints,
    can_factor_int,
    eval_ints,
    factor_int,
)
from symple.expr.actions_ring import ActionRing
from symple.expr.expr_node import ExprNode, ExprNodeType, one, zero

r = ActionRing(
    add_group=ActionGroup(
        action=ExprNodeType.ADD,
        inv=ExprNodeType.NEG,
        unit=zero,
        is_abelian=True,
    ),
        mul_group=ActionGroup(
        action=ExprNodeType.MUL,
        inv=ExprNodeType.INV,
        unit=one,
        is_abelian=True,
    ),
)

@dataclass
class Action:
    can_apply: Callable[[ExprNode], bool]
    apply: Callable[[ExprNode], ExprNode]

class ActionType(IntEnum):
    COMMUTE = 0
    ASSOCIATE_LEFT = auto()
    ASSOCIATE_RIGHT = auto()
    DISTRIBUTE_LEFT = auto()
    DISTRIBUTE_RIGHT = auto()
    FACTOR_RIGHT = auto()
    FACTOR_LEFT = auto()
    DISTRIBUTE_INV = auto()
    MULTIPLY_ZERO = auto()
    MULTIPLY_MIN_ONE = auto()
    REDUCE_DOUBLE_INV = auto()
    REDUCE_GROUP_UNIT = auto()
    ADD_ZERO = auto()
    MULTIPLY_ONE = auto()
    CANCEL = auto()
    EVAL_INTS = auto()
    FACTOR_INT = auto()


ACTIONS = {
    ActionType.COMMUTE: Action(r.can_commute, r.commute),
    ActionType.ASSOCIATE_LEFT: Action(r.can_associate_left, r.associate_left),
    ActionType.ASSOCIATE_RIGHT: Action(r.can_associate_right, r.associate_right),
    ActionType.DISTRIBUTE_LEFT: Action(r.can_distribute_left, r.distribute_left),
    ActionType.DISTRIBUTE_RIGHT: Action(r.can_distribute_right, r.distribute_right),
    ActionType.FACTOR_RIGHT: Action(r.can_factor_right, r.factor_right),
    ActionType.FACTOR_LEFT: Action(r.can_factor_left, r.factor_left),
    ActionType.DISTRIBUTE_INV: Action(r.can_distribute_inv, r.distribute_inv),
    ActionType.MULTIPLY_ZERO: Action(r.can_multiply_zero, r.multiply_zero),
    ActionType.MULTIPLY_MIN_ONE: Action(r.can_multiply_min_one, r.multiply_min_one),
    ActionType.REDUCE_DOUBLE_INV: Action(r.can_reduce_double_inv, r.reduce_double_inv),
    ActionType.CANCEL: Action(r.can_cancel, r.cancel),
    ActionType.EVAL_INTS: Action(can_eval_ints, eval_ints),
    ActionType.FACTOR_INT: Action(can_factor_int, factor_int),
    ActionType.REDUCE_GROUP_UNIT: Action(r.can_reduce_group_unit, r.reduce_group_unit),
    ActionType.ADD_ZERO: Action(r.can_add_zero, r.add_zero),
    ActionType.MULTIPLY_ONE: Action(r.can_multiply_one, r.multiply_one),
}

assert len(ACTIONS) == len(ActionType)