from collections import namedtuple

import sympy as sp
import torch

from model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE

ADD_TYPE = 0
MUL_TYPE = 1
DIV_TYPE = 2
POW_TYPE = 3
INT_NE_TYPE = 4
INT_PO_TYPE = 5 # non-negative

TYPE_LIST = [
    ADD_TYPE,
    MUL_TYPE,
    DIV_TYPE,
    POW_TYPE,
    INT_NE_TYPE,
    INT_PO_TYPE,
]

ARG_NULL = 0

Token = namedtuple('Token', ['type', 'arg'])

class Token(object):
    def __init__(self, type: int, arg: int) -> None:
        self.type = type
        self.arg = arg
    
    def to_tensor(self, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
        tensor = torch.zeros((2,), device=device, dtype=dtype)
        tensor[0] = self.type
        tensor[1] = self.arg
        return tensor

    @staticmethod
    def from_sympy(expr: sp.Expr) -> Token:
        if isinstance(expr, sp.Add):
            return Token(ADD_TYPE, ARG_NULL)
        elif isinstance(expr, sp.Mul):
            return Token(MUL_TYPE, ARG_NULL)
        elif isinstance(expr, sp.Pow):
            return Token(POW_TYPE, ARG_NULL)
        elif isinstance(expr, sp.Integer):
            return Token(INT_NE_TYPE if expr < 0 else INT_PO_TYPE, expr.args[0])
        else:
            raise NotImplementedError(f'Unsupported expression type {type(expr)}')


class BinaryTokenTree(object):
    def __init__(self, root: Token, a: Token, b: Token) -> None:
        self.root = root
        self.a = a
        self.b = b

    def to_sorted_list(self) -> list:
        if self.a is None and self.b is None:
            return [self.root]
        elif self.b is None:
            return self.a.to_list() + [self.root]
        else:
            return self.a.to_list() + self.b.to_list() + [self.root]

    @classmethod
    def from_sympy(cls, expr: sp.Expr) -> "BinaryTokenTree":
        root = Token.from_sympy(expr)
        if isinstance(expr, (sp.Add, sp.Mul)):
            a = cls.from_sympy(expr.args[0])
            b = cls.from_sympy(
                type(expr)(*expr.args[1:], evalulate=True) 
                if len(expr.args) > 2 
                else expr.args[1]
            )
            return BinaryTokenTree(
                root=root, 
                a=cls.from_sympy(a),
                b=cls.from_sympy(b),
            )
        elif isinstance(expr, sp.Pow):
            return BinaryTokenTree(
                root=root,
                a=cls.from_sympy(expr.args[0]),
                b=cls.from_sympy(expr.args[1]),
            )
        elif isinstance(expr, sp.Integer):
            return BinaryTokenTree(
                root=root, 
                a=None, 
                b=None,
            )
        else:
            raise NotImplementedError(f'Unsupported expression type {type(expr)}')
