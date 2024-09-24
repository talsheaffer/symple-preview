from symple.expr.actions_group import ActionGroup
from symple.expr.expr_node import ExprNode


class ActionRing:
    def __init__(self, add_group: ActionGroup, mul_group: ActionGroup):
        self.add = add_group.action
        self.mul = mul_group.action
        self.zero = add_group.unit
        self.one = mul_group.unit
        self.one_inv = ExprNode(mul_group.inv, left=self.one)
        self.add_group = add_group
        self.mul_group = mul_group

    def can_commute(self, expr: ExprNode) -> bool:
        return self.add_group.can_commute(expr) or self.mul_group.can_commute(expr)

    def commute(self, expr: ExprNode) -> ExprNode:
        assert self.can_commute(expr)
        if self.add_group.can_commute(expr):
            return self.add_group.commute(expr)
        return self.mul_group.commute(expr)

    def can_associate_right(self, expr: ExprNode) -> bool:
        return self.add_group.can_associate_right(
            expr
        ) or self.mul_group.can_associate_right(expr)

    def associate_right(self, expr: ExprNode) -> ExprNode:
        assert self.can_associate_right(expr)
        if self.add_group.can_associate_right(expr):
            return self.add_group.associate_right(expr)
        return self.mul_group.associate_right(expr)

    def can_associate_left(self, expr: ExprNode) -> bool:
        return self.add_group.can_associate_left(
            expr
        ) or self.mul_group.can_associate_left(expr)

    def associate_left(self, expr: ExprNode) -> ExprNode:
        assert self.can_associate_left(expr)
        if self.add_group.can_associate_left(expr):
            return self.add_group.associate_left(expr)
        return self.mul_group.associate_left(expr)

    def can_cancel(self, expr: ExprNode) -> bool:
        return self.add_group.can_cancel(expr) or self.mul_group.can_cancel(expr)

    def cancel(self, expr: ExprNode) -> ExprNode:
        assert self.can_cancel(expr)
        if self.add_group.can_cancel(expr):
            return self.add_group.cancel(expr)
        return self.mul_group.cancel(expr)

    def can_reduce_group_unit(self, expr: ExprNode) -> bool:
        return self.add_group.can_reduce_unit(expr) or self.mul_group.can_reduce_unit(expr)

    def reduce_group_unit(self, expr: ExprNode) -> ExprNode:
        assert self.can_reduce_group_unit(expr)
        if self.add_group.can_reduce_unit(expr):
            return self.add_group.reduce_unit(expr)
        return self.mul_group.reduce_unit(expr)

    def can_add_zero(self, expr: ExprNode) -> bool:
        return True

    def add_zero(self, expr: ExprNode) -> ExprNode:
        assert self.can_add_zero(expr)
        return self.add_group.induce_unit(expr)

    def can_multiply_one(self, expr: ExprNode) -> bool:
        return True

    def multiply_one(self, expr: ExprNode) -> ExprNode:
        assert self.can_multiply_one(expr)
        return self.mul_group.induce_unit(expr)

    def can_reduce_double_inv(self, expr: ExprNode) -> bool:
        return self.add_group.can_reduce_double_inv(
            expr
        ) or self.mul_group.can_reduce_double_inv(expr)

    def reduce_double_inv(self, expr: ExprNode) -> ExprNode:
        assert self.can_reduce_double_inv(expr)
        if self.add_group.can_reduce_double_inv(expr):
            return self.add_group.reduce_double_inv(expr)
        return self.mul_group.reduce_double_inv(expr)

    def can_distribute_inv(self, expr: ExprNode) -> bool:
        return self.add_group.can_distribute_inv(
            expr
        ) or self.mul_group.can_distribute_inv(expr)

    def distribute_inv(self, expr: ExprNode) -> ExprNode:
        assert self.can_distribute_inv(expr)
        if self.add_group.can_distribute_inv(expr):
            return self.add_group.distribute_inv(expr)
        return self.mul_group.distribute_inv(expr)

    def can_multiply_zero(self, expr: ExprNode) -> bool:
        return expr.type == self.mul and (
            expr.left == self.zero or expr.right == self.zero
        )

    def multiply_zero(self, expr: ExprNode) -> ExprNode:
        # 0a = a0 = 0
        assert self.can_multiply_zero(expr)
        return self.zero

    def can_multiply_min_one(self, expr: ExprNode) -> bool:
        # (-1)a = a(-1) = -a
        return expr.type == self.mul and (
            expr.left == self.one_inv or 
            expr.right == self.one_inv
        )

    def multiply_min_one(self, expr: ExprNode) -> ExprNode:
        # (-1)a = a(-1) = -a
        assert self.can_multiply_min_one(expr)
        if expr.left == self.one_inv:
            return ExprNode(type=self.add_group.inv, left=expr.right)
        else:
            return ExprNode(type=self.add_group.inv, left=expr.left)

    def can_distribute_left(self, expr: ExprNode) -> bool:
        # a(b+c) -> ab+ac
        return expr.type == self.mul and expr.right.type == self.add

    def distribute_left(self, expr: ExprNode) -> ExprNode:
        # a(b+c) -> ab+ac
        assert self.can_distribute_left(expr)
        return ExprNode(
            type=self.add,
            left=ExprNode(type=self.mul, left=expr.left, right=expr.right.left),
            right=ExprNode(type=self.mul, left=expr.left, right=expr.right.right),
        )

    def can_distribute_right(self, expr: ExprNode) -> bool:
        # (a+b)c -> ac+bc
        return expr.type == self.mul and expr.left.type == self.add

    def distribute_right(self, expr: ExprNode) -> ExprNode:
        # (a+b)c -> ac+bc
        assert self.can_distribute_right(expr)
        return ExprNode(
            type=self.add,
            left=ExprNode(type=self.mul, left=expr.left.left, right=expr.right),
            right=ExprNode(type=self.mul, left=expr.left.right, right=expr.right),
        )

    def can_factor_right(self, expr: ExprNode) -> bool:
        # ab+ac -> a(b+c)
        return (
            expr.type == self.add
            and expr.left.type == self.mul
            and expr.right.type == self.mul
            and expr.left.left == expr.right.left
        )

    def factor_right(self, expr: ExprNode) -> ExprNode:
        # ab+ac -> a(b+c)
        assert self.can_factor_right(expr)
        return ExprNode(
            type=self.mul,
            left=expr.left.left,
            right=ExprNode(type=self.add, left=expr.left.right, right=expr.right.right),
        )

    def can_factor_left(self, expr: ExprNode) -> bool:
        # ba+ca -> (b+c)a
        return (
            expr.type == self.add
            and expr.left.type == self.mul
            and expr.right.type == self.mul
            and expr.left.right == expr.right.right
        )

    def factor_left(self, expr: ExprNode) -> ExprNode:
        # ba+ca -> (b+c)a
        assert self.can_factor_left(expr)
        return ExprNode(
            type=self.mul,
            left=ExprNode(type=self.add, left=expr.left.left, right=expr.right.left),
            right=expr.left.right
        )
