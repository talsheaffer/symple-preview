import sympy as sp
import torch
from numpy import inf

from src.model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE
from src.utils import iota

it = iota()

ADD_TYPE = next(it)
SUB_TYPE = next(it)
MUL_TYPE = next(it)
DIV_TYPE = next(it)
POW_TYPE = next(it)
INT_NE_TYPE = next(it)
INT_PO_TYPE = next(it)
INT_ZERO_TYPE = next(it)
INF_TYPE = next(it)
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
    INT_ZERO_TYPE,
    INF_TYPE,
    X_TYPE,
    Y_TYPE,
    Z_TYPE,
]

SYMPY_TO_TYPE_MAP = {   
    sp.Add: ADD_TYPE,
    sp.Mul: MUL_TYPE,
    sp.Pow: POW_TYPE,
    sp.Integer: INT_NE_TYPE,
    sp.Symbol("x"): X_TYPE,
    sp.Symbol("y"): Y_TYPE,
    sp.Symbol("z"): Z_TYPE,
    sp.core.numbers.Infinity: INF_TYPE,
    sp.core.numbers.ComplexInfinity: INF_TYPE,
}

SYMPY_SYMBOL_MAP = {
    "x": X_TYPE,
    "y": Y_TYPE,
    "z": Z_TYPE,
}

ARG_NULL = 1


class ExprNode(object):
    def __init__(
        self,
        type: int,
        arg: int = ARG_NULL,
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
        if self.a is not None:
            self.a.p = self
        if self.b is not None:
            self.b.p = self
        self.p = p
        # self.embedding = embedding# if embedding is not None else torch.nn.functional.one_hot(torch.tensor([[type]]), num_classes=VOCAB_SIZE).to(DEFAULT_DEVICE, DEFAULT_DTYPE)
        self.hidden = hidden
        self.cell = cell

        if self.a is not None:
            self.a.p = self
        if self.b is not None:
            self.b.p = self

    def node_count(self, depth: int = inf) -> int:
        if depth == 0:
            return 1
        else:
            return 1 + sum(child.node_count(depth-1) for child in self.children if child is not None)
    
    def get_node(self, coord: tuple[int, ...]) -> "ExprNode":
        if not coord:
            return self
        else:
            return self.children[coord[0]].get_node(coord[1:])
    
    def apply_at_coord(self, coord: tuple[int, ...], fn: callable) -> "ExprNode":
        if not coord:
            return fn(self)
        elif coord[0] == 0:
            self.a = self.a.apply_at_coord(coord[1:], fn)
        elif coord[0] == 1:
            self.b = self.b.apply_at_coord(coord[1:], fn)
        else:
            raise ValueError(f"Invalid coordinate {coord}")
        self.ensure_parenthood()
        return self
    
    def get_coords(self)->list[tuple[int, ...]]:
        return [()] + (
            [(0,) + tup for tup in self.a.get_coords()] if self.a is not None else []
        ) + (
            [(1,) + tup for tup in self.b.get_coords()] if self.b is not None else []
        )

    @property
    def arg_hot(self):
        return self.arg * torch.nn.functional.one_hot(torch.tensor([self.type]), num_classes=VOCAB_SIZE).to(DEFAULT_DEVICE, DEFAULT_DTYPE)
    
    @property
    def children(self):
        return (self.a, self.b)

    def topological_sort(self) -> list:
        if self.a is None and self.b is None:
            return [self]
        elif self.b is None:
            return self.a.topological_sort() + [self]
        else:
            return self.a.topological_sort() + self.b.topological_sort() + [self]

    @classmethod
    def from_sympy(cls, expr: sp.Expr, **kwargs) -> "ExprNode":
        if isinstance(expr, sp.Add):
            type, arg = ADD_TYPE, ARG_NULL
        elif isinstance(expr, sp.Mul):
            type, arg = MUL_TYPE, ARG_NULL
        elif isinstance(expr, sp.Pow):
            type, arg = POW_TYPE, ARG_NULL
        elif isinstance(expr, sp.Rational):
            if isinstance(expr, sp.Integer):
                if expr == 0:
                    type, arg = INT_ZERO_TYPE, ARG_NULL
                elif expr > 0:
                    type, arg = INT_PO_TYPE, int(expr)
                else:
                    type, arg = INT_NE_TYPE, abs(int(expr))
            else:
                type, arg = DIV_TYPE, ARG_NULL
        elif isinstance(expr, sp.Symbol) and expr.name in SYMPY_SYMBOL_MAP:
            type, arg = SYMPY_SYMBOL_MAP[expr.name], ARG_NULL
        elif type(expr) in SYMPY_TO_TYPE_MAP:
            type, arg = SYMPY_TO_TYPE_MAP[type(expr)], ARG_NULL
        else:
            raise NotImplementedError(f"Unsupported expression type {type(expr)}")

        a, b = None, None
        if isinstance(expr, (sp.Add, sp.Mul)):
            a = cls.from_sympy(expr.args[0])
            b = cls.from_sympy(expr.func(*expr.args[1:], evaluate=True))
        elif isinstance(expr, sp.Pow):
            a = cls.from_sympy(expr.args[0])
            b = cls.from_sympy(expr.args[1])
        elif isinstance(expr, sp.Rational):
            if not isinstance(expr, sp.Integer):
                a = cls.from_sympy(sp.sympify(expr.p))
                b = cls.from_sympy(sp.sympify(expr.q))

        en = ExprNode(
            type=type,
            arg=arg,
            a=a,
            b=b,
            **kwargs
        )
        return en

    @classmethod
    def from_ExprNode(cls, en: "ExprNode")->"ExprNode":
        return ExprNode(
            type=en.type,
            arg=en.arg,
            a=cls.from_ExprNode(en.a) if en.a is not None else None,
            b=cls.from_ExprNode(en.b) if en.b is not None else None,
            p=en.p,
            # embedding=en.embedding.clone(),
            hidden=en.hidden.clone() if en.hidden is not None else None,
            cell=en.cell.clone() if en.cell is not None else None,
        )
    
    def reset(self)->None:
        # self.embedding = None
        self.hidden = None
        self.cell = None
        (c.reset() for c in self.children if c is not None)

    def to_tensor(self, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
        tensor = torch.zeros((2,), device=device, dtype=dtype)
        tensor[0] = self.type
        tensor[1] = self.arg
        return tensor
    
    def ensure_parenthood(self)->None:
        if self.a is not None:
            self.a.p = self
        if self.b is not None:
            self.b.p = self

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

        en = ExprNode.from_ExprNode(self)
        en.a, en.b = en.b, en.a
        return en

    def associate_b(self) -> "ExprNode":
        # Field Axiom
        if not self.can_associate_b():
            raise ValueError(f"Cannot associate {self.type}.")

        return ExprNode(
            type=self.type,
            arg=self.arg,
            a= ExprNode(
                type=self.b.type,
                arg=self.b.arg,
                a=self.a,
                b=self.b.a,
                hidden=self.b.hidden.clone() if self.b.hidden is not None else None,
                cell=self.b.cell.clone() if self.b.cell is not None else None,
            ),
            b=self.b.b,
            p=self.p,
            hidden=self.hidden.clone() if self.hidden is not None else None,
            cell=self.cell.clone() if self.cell is not None else None,
        )

    def distribute_b(self) -> "ExprNode":
        # Field Axiom
        if not self.can_distribute_b():
            raise ValueError(f"Cannot distribute {self.type}.")

        return ExprNode(
            type=self.b.type,
            arg=ARG_NULL,
            a=ExprNode(
                type=self.type,
                arg=ARG_NULL,
                a=self.a,
                b=self.b.a,
                hidden = self.b.a.hidden.clone() if self.b.a is not None else None,
                cell = self.b.a.cell.clone() if self.b.a is not None else None,
            ),
            b=ExprNode(
                type=self.type,
                arg=ARG_NULL,
                a=ExprNode.from_ExprNode(self.a), # Duplicate self.a
                b=self.b.b,
                hidden = self.b.b.hidden.clone() if self.b.b is not None else None,
                cell = self.b.b.cell.clone() if self.b.b is not None else None,
            ),
            p = self.p,
            hidden=self.hidden.clone() if self.hidden is not None else None,
            cell=self.cell.clone() if self.cell is not None else None,
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
                hidden=self.hidden.clone() if self.hidden is not None else None,
                cell=self.cell.clone() if self.cell is not None else None,
            ),
            p=self.p,
            hidden=self.a.hidden.clone() if self.a.hidden is not None else None,
            cell=self.a.cell.clone() if self.a.cell is not None else None,
        )

    def reduce_unit(self) -> "ExprNode":
        # Field Axiom
        if not self.can_reduce_unit():
            raise ValueError(f"Cannot reduce unit {self.type}.")

        a, b = self.a, self.b

        if self.type == ADD_TYPE:
            en = b if a == zero else a
        elif self.type == MUL_TYPE:
            en = b if a == one else a
        elif self.type == POW_TYPE: # TODO: handle infinities
            if b == zero:
                en = one
            elif b == one:
                en = self.a
            elif a == zero:
                if b.type == INT_PO_TYPE:
                    en = zero
                elif b.type == INT_NE_TYPE:
                    en = inf_node
            elif a == one:
                en = one
            else:
                raise ValueError(f"Cannot reduce unit {self.type}.")
            en.hidden = self.hidden.clone() if self.hidden is not None else None
            en.cell = self.cell.clone() if self.cell is not None else None
        
        en.p = self.p
        en.ensure_parenthood()

        return en

    def cancel(self) -> "ExprNode":
        # Field Property
        if not self.can_cancel():
            raise ValueError(f"Cannot cancel {self.type}.")

        en = zero
        en.p = self.p
        en.hidden = self.hidden.clone() if self.hidden is not None else None
        en.cell = self.cell.clone() if self.cell is not None else None
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
        elif self.type == INT_ZERO_TYPE:
            return "0"
        elif self.type in [X_TYPE, Y_TYPE, Z_TYPE]:
            return {
                X_TYPE: "x",
                Y_TYPE: "y",
                Z_TYPE: "z",
            }[self.type]
        else:
            return "Invalid expression type"


one = ExprNode(INT_PO_TYPE)
minus_one = ExprNode(INT_NE_TYPE)
zero = ExprNode(INT_ZERO_TYPE)
inf_node = ExprNode(INF_TYPE)

