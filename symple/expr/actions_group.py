from symple.expr.expr_node import ExprNode, ExprNodeType


class ActionGroup:
    def __init__(self, action: ExprNodeType, inv: ExprNodeType, unit: ExprNode, is_abelian: bool):
        self.action = action
        self.inv = inv
        self.unit = unit
        self.is_abelian = is_abelian

    def can_commute(self, expr: ExprNode) -> bool:
        # ab -> ba
        return expr.type == self.action and self.is_abelian

    def commute(self, expr: ExprNode) -> ExprNode:
        # ab -> ba
        assert self.can_commute(expr)
        return ExprNode(type=self.action, left=expr.right, right=expr.left)

    def can_associate_right(self, expr: ExprNode) -> bool:
        # a(bc) -> (ab)c
        return expr.type == self.action and expr.right.type == self.action

    def associate_right(self, expr: ExprNode) -> ExprNode:
        # a(bc) -> (ab)c
        assert self.can_associate_right(expr)
        return ExprNode(
            type=self.action,
            left=ExprNode(type=self.action, left=expr.left, right=expr.right.left),
            right=expr.right.right,
        )

    def can_associate_left(self, expr: ExprNode) -> bool:
        # (ab)c -> a(bc)
        return expr.type == self.action and expr.left.type == self.action

    def associate_left(self, expr: ExprNode) -> ExprNode:
        # (ab)c -> a(bc)
        assert self.can_associate_left(expr)
        return ExprNode(
            type=self.action,
            left=expr.left.left,
            right=ExprNode(type=self.action, left=expr.left.right, right=expr.right),
        )

    def can_cancel(self, expr: ExprNode) -> bool:
        return (
            # (a^-1)a -> e
            expr.type == self.action
            and expr.left.type == self.inv
            and expr.right == expr.left.left
        ) or (
            # a(a^-1) -> e
            expr.type == self.action
            and expr.right.type == self.inv
            and expr.right.left == expr.left
        )

    def cancel(self, expr: ExprNode) -> ExprNode:
        # (a^-1)a -> e or a(a^-1) -> e
        assert self.can_cancel(expr)
        return self.unit

    def can_reduce_unit(self, expr: ExprNode) -> bool:
        return expr.type == self.action and (
            expr.left == self.unit
            or expr.right == self.unit
        )

    def reduce_unit(self, expr: ExprNode) -> ExprNode:
        assert self.can_reduce_unit(expr)
        if expr.left == self.unit:
            return expr.right
        return expr.left

    def can_induce_unit(self, expr: ExprNode) -> bool:
        return True

    def induce_unit(self, expr: ExprNode) -> ExprNode:
        assert self.can_induce_unit(expr)
        # shallow clone to avoid cycle when replacing expr
        return ExprNode(type=self.action, left=self.unit, right=expr.shallow_clone())

    def can_reduce_double_inv(self, expr: ExprNode) -> bool:
        # (a^-1)^-1 -> a
        return expr.type == self.inv and expr.left.type == self.inv

    def reduce_double_inv(self, expr: ExprNode) -> ExprNode:
        # (a^-1)^-1 -> a
        assert self.can_reduce_double_inv(expr)
        return expr.left.left

    def can_induce_double_inv(self, expr: ExprNode) -> bool:
        # a -> (a^-1)^-1
        return True

    def induce_double_inv(self, expr: ExprNode) -> ExprNode:
        # a -> (a^-1)^-1
        assert self.can_induce_double_inv(expr)
        return ExprNode(type=self.inv, left=ExprNode(type=self.inv, left=expr))

    def can_distribute_inv(self, expr: ExprNode) -> bool:
        # (ab)^-1 -> b^-1 a^-1
        return expr.type == self.inv and expr.left.type == self.action

    def distribute_inv(self, expr: ExprNode) -> ExprNode:
        # (ab)^-1 -> b^-1 a^-1
        assert self.can_distribute_inv(expr)
        return ExprNode(
            type=self.action,
            left=ExprNode(type=self.inv, left=expr.left.right),
            right=ExprNode(type=self.inv, left=expr.left.left),
        )
