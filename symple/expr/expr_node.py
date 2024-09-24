from enum import IntEnum, auto
from typing import Optional, Union

import sympy as sp


class ExprNodeType(IntEnum):
    ADD = 0
    NEG = auto()
    MUL = auto()
    INV = auto()
    POW = auto()
    INT = auto()
    VAR_X = auto()
    VAR_Y = auto()
    VAR_Z = auto()


ARG_NULL = 0


class ExprNode(object):
    def __init__(
        self,
        type: ExprNodeType,
        arg: Optional[int] = ARG_NULL,
        left: Optional["ExprNode"] = None,
        right: Optional["ExprNode"] = None,
    ) -> None:
        self.type = type
        self.arg = arg
        self.left = left
        self.right = right

    def replace(self, other: "ExprNode") -> None:
        self.type = other.type
        self.arg = other.arg
        self.left = other.left
        self.right = other.right

    def shallow_clone(self) -> "ExprNode":
        return ExprNode(self.type, self.arg, self.left, self.right)

    def clone(self) -> "ExprNode":
        expr = ExprNode(self.type, self.arg)
        if self.left is not None:
            expr.left = self.left.clone()
        if self.right is not None:
            expr.right = self.right.clone()
        return expr

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children())

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def children(self):
        return [x for x in [self.left, self.right] if x is not None]

    def topological_sort(self) -> list:
        if self.left is None and self.right is None:
            return [self]
        elif self.right is None:
            return self.left.topological_sort() + [self]
        elif self.left is None:
            return self.right.topological_sort() + [self]
        else:
            return self.left.topological_sort() + self.right.topological_sort() + [self]

    @classmethod
    def from_sympy(cls, expr: Union[sp.Expr, str, int]) -> "ExprNode":
        if isinstance(expr, str):
            expr = sp.sympify(expr, evaluate=False)

        if isinstance(expr, sp.Add):
            left = cls.from_sympy(expr.args[0])
            right = cls.from_sympy(sp.Add(*expr.args[1:], evaluate=False))
            return ExprNode(ExprNodeType.ADD, left=left, right=right)

        if isinstance(expr, sp.Mul):
            left = cls.from_sympy(expr.args[0])
            right = cls.from_sympy(sp.Mul(*expr.args[1:], evaluate=False))
            if left.type == ExprNodeType.INT:
                if left.arg == 1:
                    return right
                elif left.arg == -1:
                    return ExprNode(ExprNodeType.NEG, left=right)
            return ExprNode(ExprNodeType.MUL, left=left, right=right)

        if isinstance(expr, sp.Pow):
            if expr.exp == -1:
                return ExprNode(ExprNodeType.INV, left=cls.from_sympy(expr.base))
            
            left = cls.from_sympy(expr.base)
            right = cls.from_sympy(expr.exp)
            return ExprNode(ExprNodeType.POW, left=left, right=right)

        if isinstance(expr, (sp.Rational, int)):
            if isinstance(expr, (sp.Integer, int)):
                expr = ExprNode(ExprNodeType.INT, int(expr))
                if expr.arg < 0:
                    expr.arg = -expr.arg
                    return ExprNode(ExprNodeType.NEG, left=expr)
                return expr
            
            left = cls.from_sympy(expr.p)
            right = cls.from_sympy(expr.q)
            right_inv = ExprNode(ExprNodeType.INV, left=right)
            return ExprNode(ExprNodeType.MUL, left=left, right=right_inv)
        
        if isinstance(expr, sp.Symbol):
            var_type = 'VAR_' + expr.name.upper()
            return ExprNode(ExprNodeType[var_type])
        
        raise NotImplementedError(f"Unsupported expression type {type(expr)}")

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, ExprNode)
            and self.type == value.type
            and self.arg == value.arg
            and self.left == value.left
            and self.right == value.right
        )

    def __repr__(self) -> str:
        if self.type == ExprNodeType.INT:
            return str(self.arg)
        
        if self.type == ExprNodeType.VAR_X:
            return "x"
        
        if self.type == ExprNodeType.VAR_Y:
            return "y"
        
        if self.type == ExprNodeType.VAR_Z:
            return "z"
        
        if self.type == ExprNodeType.ADD:
            return f"({str(self.left)} + {str(self.right)})"
        
        if self.type == ExprNodeType.MUL:
            return f"({str(self.left)} * {str(self.right)})"
        
        if self.type == ExprNodeType.NEG:
            return f"(-{str(self.left)})"
        
        if self.type == ExprNodeType.INV:
            return f"(1/{str(self.left)})"
        
        if self.type == ExprNodeType.POW:
            return f"({str(self.left)}^{str(self.right)})"
        
        raise NotImplementedError(f"Unsupported expression type {self.type}")


one = ExprNode(ExprNodeType.INT, 1)
zero = ExprNode(ExprNodeType.INT, 0)
one_inv = ExprNode(ExprNodeType.INV, left=one)