from src.model.tree import (ADD_TYPE, ARG_NULL, INT_PO_TYPE, MUL_TYPE,
                            POW_TYPE, SUB_TYPE, ExprNode)


class FieldNode(ExprNode):
    def can_commute(self) -> bool:
        # Field Axiom
        return self.type in (ADD_TYPE, MUL_TYPE)

    def can_distribute_b(self) -> bool:
        # Field Axiom
        return (
            self.type == MUL_TYPE and
            self.b.type == ADD_TYPE
        ) or (
            self.type == POW_TYPE and
            self.b.type == MUL_TYPE
        )
    
    def can_undistribute_b(self) -> bool:
        # Field Axiom
        return (
            self.type == ADD_TYPE and
            self.a.type == MUL_TYPE and
            self.b.type == MUL_TYPE and
            self.a.a == self.b.a
        ) or (
            self.type == MUL_TYPE and
            self.a.type == POW_TYPE and
            self.b.type == POW_TYPE and
            self.a.a == self.b.a
        )
    
    def can_reduce_unit(self) -> bool:
        # Field Axiom
        return (
            (
                self.type == ADD_TYPE and
                self.a.arg == 0 or self.b.arg == 0
            ) or (
                self.type == MUL_TYPE and
                self.a.arg == 1 or self.b.arg == 1
            ) or (
                self.type == POW_TYPE and
                len({0, 1} & {self.a.arg, self.b.arg}) > 0
            )
        )
    
    def can_eliminate(self) -> bool:
        # Field Axiom
        return (
            self.type == SUB_TYPE and
            self.a == self.b
        ) or (
            self.type == ADD_TYPE and
            self.a.arg == 0 or self.b.arg == 0
        )
    
    def commute(self) -> "FieldNode":
        # Field Axiom
        if not self.can_commute():
            raise ValueError(f"Cannot commute {self.type}.")

        return ExprNode(
            root=ExprNode(self.type, ARG_NULL),
            a=self.b,
            b=self.a,
        )

    def distribute_b(self) -> "FieldNode":
        # Field Axiom
        if not self.can_distribute_b():
            raise ValueError(f"Cannot distribute {self.root.type}.")

        r, a, b = self.root, self.a, self.b
        return ExprNode(
            root=ExprNode(a.type, ARG_NULL),
            a=ExprNode(
                root=ExprNode(r.type, ARG_NULL),
                a=a,
                b=a.b,
            ),
            b=ExprNode(
                root=ExprNode(r.type, ARG_NULL),
                a=a,
                b=b.b,
            ),
        )
    
    def undistribute_b(self) -> "FieldNode":
        # Field Axiom
        if not self.can_undistribute_b():
            raise ValueError(f"Cannot undistribute {self.root.type}.")

        r, a, b = self.root, self.a, self.b
        return ExprNode(
            root=ExprNode(a.type, ARG_NULL),
            a=a.a,
            b=ExprNode(
                root=ExprNode(r.type, ARG_NULL),
                a=a.b,
                b=b.b,
            ),
        )
    
    def reduce_unit(self) -> "FieldNode":
        # Field Axiom
        if not self.can_reduce_unit():
            raise ValueError(f"Cannot reduce unit {self.root.type}.")

        r, a, b = self.root, self.a, self.b
        if self.type == ADD_TYPE:
            return a if a.arg == 0 else b
        elif self.type == MUL_TYPE:
            return a if a.arg == 1 else b
        # self.type == POW_TYPE:
        if b.arg == 1:
            return a
        # either b.arg == 0 or a.arg in (0, 1)
        return ExprNode(INT_PO_TYPE, 1)

    def eliminate(self) -> "FieldNode":
        # Field Axiom
        if not self.can_eliminate():
            raise ValueError(f"Cannot eliminate {self.root.type}.")
        
        return ExprNode(INT_PO_TYPE, 0)
