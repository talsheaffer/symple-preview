import sympy as sp
import torch

from src.model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE
from src.utils import iota

it = iota()

ADD_TYPE = next(it)
SUB_TYPE = next(it)
MUL_TYPE = next(it)
DIV_TYPE = next(it)
POW_TYPE = next(it)
INT_NE_TYPE = next(it)
INT_PO_TYPE = next(it)  # non-negative
X_TYPE = next(it)
Y_TYPE = next(it)
Z_TYPE = next(it)

VOCAB_SIZE = next(it)

TYPE_LIST = [
    ADD_TYPE,
    SUB_TYPE,
    MUL_TYPE,
    DIV_TYPE,
    POW_TYPE,
    INT_NE_TYPE,
    INT_PO_TYPE,
    X_TYPE,
    Y_TYPE,
    Z_TYPE,
]

SYMPY_TO_TYPE_MAP = (
    {
        sp.Add: ADD_TYPE,
        sp.Mul: MUL_TYPE,
        sp.Pow: POW_TYPE,
        sp.Integer: INT_NE_TYPE,
        sp.Symbol("x"): X_TYPE,
        sp.Symbol("y"): Y_TYPE,
        sp.Symbol("z"): Z_TYPE,
    },
)

SYMPY_SYMBOL_MAP = {
    "x": X_TYPE,
    "y": Y_TYPE,
    "z": Z_TYPE,
}

ARG_NULL = 0


class ExprNode(object):
    def __init__(
        self,
        type: int,
        arg: int,
        a: "ExprNode" = None,
        b: "ExprNode" = None,
        p: "ExprNode" = None,
        embedding: torch.Tensor = None,
        hidden: torch.Tensor = None,
        cell: torch.Tensor = None,
    ) -> None:
        self.type = type
        self.arg = arg
        self.a = a
        self.b = b
        self.p = p
        self.embedding = embedding
        self.hidden = hidden
        self.cell = cell

        if self.a is not None:
            self.a.p = self
        if self.b is not None:
            self.b.p = self

    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children)

    @property
    def children(self):
        return [x for x in [self.a, self.b] if x is not None]

    def topological_sort(self) -> list:
        if self.a is None and self.b is None:
            return [self]
        elif self.b is None:
            return self.a.topological_sort() + [self]
        else:
            return self.a.topological_sort() + self.b.topological_sort() + [self]

    @classmethod
    def from_sympy(cls, expr: sp.Expr, p: "ExprNode" = None) -> "ExprNode":
        if isinstance(expr, sp.Add):
            type_, arg = ADD_TYPE, ARG_NULL
        elif isinstance(expr, sp.Mul):
            type_, arg = MUL_TYPE, ARG_NULL
        elif isinstance(expr, sp.Pow):
            type_, arg = POW_TYPE, ARG_NULL
        elif isinstance(expr, sp.Number):
            if isinstance(expr, sp.Integer):
                type_, arg = INT_NE_TYPE if expr < 0 else INT_PO_TYPE, abs(int(expr))
            elif isinstance(expr, sp.Rational):
                type_, arg = DIV_TYPE, ARG_NULL
            else:
                raise NotImplementedError(f"Unsupported number type {type(expr)}")
        elif isinstance(expr, sp.Symbol) and expr.name in SYMPY_SYMBOL_MAP:
            type_, arg = SYMPY_SYMBOL_MAP[expr.name], ARG_NULL
        else:
            raise NotImplementedError(f"Unsupported expression type {type(expr)}")

        en = ExprNode(
            type=type_,
            arg=arg,
        )
        if isinstance(expr, (sp.Add, sp.Mul)):
            en.a = cls.from_sympy(expr.args[0], p=en)
            en.b = cls.from_sympy(expr.func(*expr.args[1:], evaluate=True), p=en)
        elif isinstance(expr, sp.Pow):
            en.a = cls.from_sympy(expr.args[0], p=en)
            en.b = cls.from_sympy(expr.args[1], p=en)
        elif isinstance(expr, sp.Rational):
            en.a = cls.from_sympy(expr.p, p=en)
            en.b = cls.from_sympy(expr.q, p=en)

        return en

    def to_tensor(self, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
        tensor = torch.zeros((2,), device=device, dtype=dtype)
        tensor[0] = self.type
        tensor[1] = self.arg
        return tensor

    def can_commute(self) -> bool:
        # Field Axiom
        return self.type in (ADD_TYPE, MUL_TYPE)

    def can_associate_b(self) -> bool:
        # Field Axiom
        return (self.type == ADD_TYPE and self.b.type == ADD_TYPE) or (
            self.type == MUL_TYPE and self.b.type == MUL_TYPE
        )

    def can_distribute_b(self) -> bool:
        # Field Axiom
        return (self.type == MUL_TYPE and self.b.type == ADD_TYPE) or (
            self.type == POW_TYPE and self.b.type == MUL_TYPE
        )

    def can_undistribute_b(self) -> bool:
        # Field Axiom
        return (
            self.type == ADD_TYPE
            and self.a.type == MUL_TYPE
            and self.b.type == MUL_TYPE
            and self.a.a == self.b.a
        ) or (
            self.type == MUL_TYPE
            and self.a.type == POW_TYPE
            and self.b.type == POW_TYPE
            and self.a.a == self.b.a
        )

    def can_reduce_unit(self) -> bool:
        # Field Axiom
        return (
            (self.type == ADD_TYPE and self.a == zero or self.b == zero)
            or (self.type == MUL_TYPE and self.a == one or self.b == one)
            or (
                self.type == POW_TYPE
                and (self.b == zero or self.b == one or self.a == zero or self.a == one)
            )
        )

    def can_cancel(self) -> bool:
        # Field Property
        if self.type != ADD_TYPE:
            return False

        def can_cancel_b(a, b):
            # expect b = (-1) * b.b or b = b.b * (-1)
            if b.type != MUL_TYPE:
                return False
            ba, bb = b.a, b.b
            if b.a != minus_one:
                ba, bb = bb, ba
            return a == bb

        return can_cancel_b(self.a, self.b) or can_cancel_b(self.b, self.a)

    def commute(self) -> "ExprNode":
        # Field Axiom
        if not self.can_commute():
            raise ValueError(f"Cannot commute {self.type}.")

        return ExprNode(
            type=self.type,
            arg=ARG_NULL,
            a=self.b,
            b=self.a,
        )

    def associate_b(self) -> "ExprNode":
        # Field Axiom
        if not self.can_associate_b():
            raise ValueError(f"Cannot associate {self.type}.")

        a, b = self.a, self.b
        return ExprNode(
            type=self.type,
            arg=ARG_NULL,
            a=ExprNode(
                type=self.type,
                arg=ARG_NULL,
                a=a,
                b=b.a,
            ),
            b=b.b,
        )

    def distribute_b(self) -> "ExprNode":
        # Field Axiom
        if not self.can_distribute_b():
            raise ValueError(f"Cannot distribute {self.type}.")

        a, b = self.a, self.b
        return ExprNode(
            type=b.type,
            arg=ARG_NULL,
            a=ExprNode(
                type=self.type,
                arg=ARG_NULL,
                a=a,
                b=b.a,
            ),
            b=ExprNode(
                type=self.type,
                arg=ARG_NULL,
                a=a,
                b=b.b,
            ),
        )

    def undistribute_b(self) -> "ExprNode":
        # Field Axiom
        if not self.can_undistribute_b():
            raise ValueError(f"Cannot undistribute {self.type}.")

        a, b = self.a, self.b
        return ExprNode(
            type=a.type,
            arg=ARG_NULL,
            a=a.a,
            b=ExprNode(
                type=self.type,
                arg=ARG_NULL,
                a=a.b,
                b=b.b,
            ),
        )

    def reduce_unit(self) -> "ExprNode":
        # Field Axiom
        if not self.can_reduce_unit():
            raise ValueError(f"Cannot reduce unit {self.type}.")

        a, b = self.a, self.b
        if self.type == ADD_TYPE:
            return b if a == zero else a
        elif self.type == MUL_TYPE:
            return b if a == one else a
        elif self.type == POW_TYPE:
            if b == zero:
                return one
            elif b == one:
                return self.a
            elif a == zero:
                return zero
            elif a == one:
                return one

        raise ImportError("Something went wrong")

    def cancel(self) -> "ExprNode":
        # Field Property
        if not self.can_cancel():
            raise ValueError(f"Cannot cancel {self.type}.")

        return zero

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, ExprNode)
            and self.type == value.type
            and self.arg == value.arg
            and self.a == value.a
            and self.b == value.b
        )

    def __repr__(self) -> str:
        if self.type in [ADD_TYPE, SUB_TYPE, MUL_TYPE, DIV_TYPE, POW_TYPE]:
            operator = {
                ADD_TYPE: "+",
                SUB_TYPE: "-",
                MUL_TYPE: "*",
                DIV_TYPE: "/",
                POW_TYPE: "^",
            }[self.type]
            return f"({str(self.a)} {operator} {str(self.b)})"
        elif self.type == INT_NE_TYPE:
            return f"(-{self.arg})"
        elif self.type == INT_PO_TYPE:
            return f"{self.arg}"
        elif self.type in [X_TYPE, Y_TYPE, Z_TYPE]:
            return {
                X_TYPE: "x",
                Y_TYPE: "y",
                Z_TYPE: "z",
            }[self.type]
        else:
            return "Invalid expression type"


one = ExprNode(INT_PO_TYPE, 1)
minus_one = ExprNode(INT_NE_TYPE, 1)
zero = ExprNode(INT_PO_TYPE, 0)
