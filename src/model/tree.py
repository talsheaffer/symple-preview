from collections import namedtuple
from typing import Tuple

import sympy as sp
import torch

from src.model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE

ADD_TYPE = 0
SUB_TYPE = 1
MUL_TYPE = 2
DIV_TYPE = 3
POW_TYPE = 4
INT_NE_TYPE = 100
INT_PO_TYPE = 101 # non-negative
X_TYPE = 200
Y_TYPE = 201
Z_TYPE = 202

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

SYMPY_TO_TYPE_MAP = {
    sp.Add: ADD_TYPE,
    sp.Mul: MUL_TYPE,
    sp.Pow: POW_TYPE,
    sp.Integer: INT_NE_TYPE,
    sp.Symbol('x'): X_TYPE,
    sp.Symbol('y'): Y_TYPE,
    sp.Symbol('z'): Z_TYPE,
},

SYMPY_SYMBOL_MAP = {
    'x': X_TYPE,
    'y': Y_TYPE,
    'z': Z_TYPE,
}

ARG_NULL = 0        

class ExprNode(object):
    def __init__(
            self, type: int, arg: int,
            a: "ExprNode" = None, b: "ExprNode" = None, p: "ExprNode" = None,
            embedding: "torch.tensor" = None, hidden: "torch.tensor" = None,
            cell: "torch.tensor" = None,
    ) -> None:
        self.type = type
        self.arg = arg
        self.a = a
        self.b = b
        self.p = p
        self.embedding = embedding
        self.hidden = hidden
        self.cell = cell

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
    def from_sympy(cls, expr: sp.Expr) -> "ExprNode":
        type_, arg = None, None

        if isinstance(expr, sp.Add):
            type_, arg = ADD_TYPE, ARG_NULL
        elif isinstance(expr, sp.Mul):
            type_, arg = MUL_TYPE, ARG_NULL
        elif isinstance(expr, sp.Pow):
            type_, arg = POW_TYPE, ARG_NULL
        elif isinstance(expr, sp.Integer):
            type_, arg = INT_NE_TYPE if expr < 0 else INT_PO_TYPE, abs(int(expr))
        elif isinstance(expr, sp.Symbol) and expr.name in SYMPY_SYMBOL_MAP:
            type_, arg = SYMPY_SYMBOL_MAP[expr.name], ARG_NULL
        else:
            raise NotImplementedError(f'Unsupported expression type {type(expr)}')

        if isinstance(expr, (sp.Add, sp.Mul)):
            a = cls.from_sympy(expr.args[0])
            b = cls.from_sympy(
                expr.func(*expr.args[1:], evaluate=True)
                # if len(expr.args) > 2
                # else expr.args[1]
            )
        elif isinstance(expr, sp.Pow):
            a=cls.from_sympy(expr.args[0])
            b=cls.from_sympy(expr.args[1])
        else:
            a = None
            b = None
        return ExprNode(
            type =type_,
            arg=arg,
            a=a,
            b=b,
        )

    def to_tensor(self, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
        tensor = torch.zeros((2,), device=device, dtype=dtype)
        tensor[0] = self.type
        tensor[1] = self.arg
        return tensor

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, ExprNode) and 
            self.type == value.type and 
            self.arg == value.arg and
            self.a == value.a and 
            self.b == value.b
        )
