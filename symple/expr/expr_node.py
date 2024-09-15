from enum import IntEnum, auto
from typing import Optional

import sympy as sp


class ExprNodeType(IntEnum):
    ADD_TYPE = auto()
    SUB_TYPE = auto()
    MUL_TYPE = auto()
    DIV_TYPE = auto()
    POW_TYPE = auto()
    INT_NE_TYPE = auto()
    INT_PO_TYPE = auto()  # non-negative
    X_TYPE = auto()
    Y_TYPE = auto()
    Z_TYPE = auto()


TYPE_OPERATOR = [
    ExprNodeType.ADD_TYPE,
    ExprNodeType.SUB_TYPE,
    ExprNodeType.MUL_TYPE,
    ExprNodeType.DIV_TYPE,
    ExprNodeType.POW_TYPE,
]

TYPE_LEAF = [
    ExprNodeType.INT_NE_TYPE,
    ExprNodeType.INT_PO_TYPE,
    ExprNodeType.X_TYPE,
    ExprNodeType.Y_TYPE,
    ExprNodeType.Z_TYPE,
]

SYMPY_TO_TYPE_MAP = (
    {
        sp.Add: ExprNodeType.ADD_TYPE,
        sp.Mul: ExprNodeType.MUL_TYPE,
        sp.Pow: ExprNodeType.POW_TYPE,
        sp.Integer: ExprNodeType.INT_NE_TYPE,
        sp.Symbol("x"): ExprNodeType.X_TYPE,
        sp.Symbol("y"): ExprNodeType.Y_TYPE,
        sp.Symbol("z"): ExprNodeType.Z_TYPE,
    },
)

SYMPY_SYMBOL_MAP = {
    "x": ExprNodeType.X_TYPE,
    "y": ExprNodeType.Y_TYPE,
    "z": ExprNodeType.Z_TYPE,
}

ARG_NULL = 0


class ExprNode(object):
    def __init__(
        self,
        type: int,
        arg: int,
        a: Optional["ExprNode"] = None,
        b: Optional["ExprNode"] = None,
        p: Optional["ExprNode"] = None,
    ) -> None:
        self.type = type
        self.arg = arg
        self.a = a
        self.b = b
        self.p = p
        
        if self.a is not None:
            self.a.p = self
        if self.b is not None:
            self.b.p = self

    def replace(self, other: "ExprNode") -> None:
        self.type = other.type
        self.arg = other.arg
        self.a = other.a
        self.b = other.b
        self.p = other.p

    def clone(self) -> "ExprNode":
        expr = ExprNode(self.type, self.arg)
        if self.a is not None:
            expr.a = self.a.clone()
            expr.a.p = expr
        if self.b is not None:
            expr.b = self.b.clone()
            expr.b.p = expr
        return expr

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children())

    def is_leaf(self) -> bool:
        return self.a is None and self.b is None

    def children(self):
        return [x for x in [self.a, self.b] if x is not None]

    def topological_sort(self) -> list:
        if self.a is None and self.b is None:
            return [self]
        elif self.b is None:
            return self.a.topological_sort() + [self]
        elif self.a is None:
            return self.b.topological_sort() + [self]
        else:
            return self.a.topological_sort() + self.b.topological_sort() + [self]

    @classmethod
    def from_sympy_str(cls, expr: str) -> "ExprNode":
        return cls.from_sympy(sp.sympify(expr, evaluate=False))

    @classmethod
    def from_sympy(cls, expr: sp.Expr) -> "ExprNode":
        if isinstance(expr, sp.Add):
                type_, arg = ExprNodeType.ADD_TYPE, ARG_NULL
        elif isinstance(expr, sp.Mul):
            type_, arg = ExprNodeType.MUL_TYPE, ARG_NULL
        elif isinstance(expr, sp.Pow):
            type_, arg = ExprNodeType.POW_TYPE, ARG_NULL
        elif isinstance(expr, sp.Rational):
            if isinstance(expr, sp.Integer):
                type_, arg = ExprNodeType.INT_NE_TYPE if expr < 0 else ExprNodeType.INT_PO_TYPE, abs(int(expr))
            else:
                type_, arg = ExprNodeType.DIV_TYPE, ARG_NULL
        elif isinstance(expr, sp.Symbol) and expr.name in SYMPY_SYMBOL_MAP:
            type_, arg = SYMPY_SYMBOL_MAP[expr.name], ARG_NULL
        else:
            raise NotImplementedError(f"Unsupported expression type {type(expr)}")

        en = ExprNode(
            type=type_,
            arg=arg,
        )
        if isinstance(expr, (sp.Add, sp.Mul)):
            en.a = cls.from_sympy(expr.args[0])
            en.b = cls.from_sympy(expr.func(*expr.args[1:], evaluate=True))
        elif isinstance(expr, sp.Pow):
            en.a = cls.from_sympy(expr.args[0])
            en.b = cls.from_sympy(expr.args[1])
        elif isinstance(expr, sp.Rational):
            if not isinstance(expr, sp.Integer):
                en.a = cls.from_sympy(sp.sympify(expr.p))
                en.b = cls.from_sympy(sp.sympify(expr.q))

        if en.a is not None:
            en.a.p = en
        if en.b is not None:
            en.b.p = en

        return en

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, ExprNode)
            and self.type == value.type
            and self.arg == value.arg
            and self.a == value.a
            and self.b == value.b
        )

    def __repr__(self) -> str:
        if self.type in TYPE_OPERATOR:
            operator = {
                ExprNodeType.ADD_TYPE: "+",
                ExprNodeType.SUB_TYPE: "-",
                ExprNodeType.MUL_TYPE: "*",
                ExprNodeType.DIV_TYPE: "/",
                ExprNodeType.POW_TYPE: "^",
            }[self.type]
            return f"({str(self.a)} {operator} {str(self.b)})"
        elif self.type == ExprNodeType.INT_NE_TYPE:
            return f"(-{self.arg})"
        elif self.type == ExprNodeType.INT_PO_TYPE:
            return f"{self.arg}"
        elif self.type in [ExprNodeType.X_TYPE, ExprNodeType.Y_TYPE, ExprNodeType.Z_TYPE]:
            return {
                ExprNodeType.X_TYPE: "x",
                ExprNodeType.Y_TYPE: "y",
                ExprNodeType.Z_TYPE: "z",
            }[self.type]
        else:
            return "Invalid expression type"


one = ExprNode(ExprNodeType.INT_PO_TYPE, 1)
minus_one = ExprNode(ExprNodeType.INT_NE_TYPE, 1)
zero = ExprNode(ExprNodeType.INT_PO_TYPE, 0)
