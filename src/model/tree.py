from collections import namedtuple

import sympy as sp
import torch

from model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE

ADD_TYPE = 0
MUL_TYPE = 1
DIV_TYPE = 2
POW_TYPE = 3
INT_NE_TYPE = 100
INT_PO_TYPE = 101 # non-negative
X_TYPE = 200
Y_TYPE = 201
Z_TYPE = 202

TYPE_LIST = [
    ADD_TYPE,
    MUL_TYPE,
    DIV_TYPE,
    POW_TYPE,
    INT_NE_TYPE,
    INT_PO_TYPE,
    X_TYPE,
    Y_TYPE,
    Z_TYPE,
]

SYMPY_SYMBOL_MAP = {
    'x': X_TYPE,
    'y': Y_TYPE,
    'z': Z_TYPE,
}

ARG_NULL = 0

ExprNode = namedtuple('Node', ['type', 'arg'])

class ExprNode(object):
    def __init__(self, type: int, arg: int) -> None:
        self.type = type
        self.arg = arg
    
    def to_tensor(self, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE) -> torch.Tensor:
        tensor = torch.zeros((2,), device=device, dtype=dtype)
        tensor[0] = self.type
        tensor[1] = self.arg
        return tensor

    @staticmethod
    def from_sympy(expr: sp.Expr) -> ExprNode:
        if isinstance(expr, sp.Add):
            return ExprNode(ADD_TYPE, ARG_NULL)
        elif isinstance(expr, sp.Mul):
            return ExprNode(MUL_TYPE, ARG_NULL)
        elif isinstance(expr, sp.Pow):
            return ExprNode(POW_TYPE, ARG_NULL)
        elif isinstance(expr, sp.Integer):
            return ExprNode(INT_NE_TYPE if expr < 0 else INT_PO_TYPE, int(expr))
        elif isinstance(expr, sp.Symbol) and expr.name in SYMPY_SYMBOL_MAP:
            return ExprNode(SYMPY_SYMBOL_MAP[expr.name], ARG_NULL)
        else:
            raise NotImplementedError(f'Unsupported expression type {type(expr)}')


class ExprBinaryTree(object):
    def __init__(self, root: ExprNode, a: ExprNode, b: ExprNode) -> None:
        self.root = root
        self.a = a
        self.b = b

    def expand_b(self) -> "ExprBinaryTree":
        r = self.root
        
        if r.type != MUL_TYPE:
            return self
        a, b = self.a, self.b
        
        if b.type != ADD_TYPE:
            return self
        ba, bb = b.a, b.b

        return ExprBinaryTree(
            root=ExprNode(ADD_TYPE, ARG_NULL),
            a=ExprBinaryTree(
                root=ExprNode(MUL_TYPE, ARG_NULL),
                a=a,
                b=ba,
            ),
            b=ExprBinaryTree(
                root=ExprNode(MUL_TYPE, ARG_NULL),
                a=a,
                b=bb,
            ),
        )

    def topological_sort(self) -> list:
        if self.a is None and self.b is None:
            return [self.root]
        elif self.b is None:
            return self.a.to_list() + [self.root]
        else:
            return self.a.to_list() + self.b.to_list() + [self.root]

    @classmethod
    def from_sympy(cls, expr: sp.Expr) -> "ExprBinaryTree":
        root = ExprNode.from_sympy(expr)
        if isinstance(expr, (sp.Add, sp.Mul)):
            a = cls.from_sympy(expr.args[0])
            b = cls.from_sympy(
                type(expr)(*expr.args[1:], evalulate=True) 
                if len(expr.args) > 2 
                else expr.args[1]
            )
            return ExprBinaryTree(
                root=root, 
                a=cls.from_sympy(a),
                b=cls.from_sympy(b),
            )
        elif isinstance(expr, sp.Pow):
            return ExprBinaryTree(
                root=root,
                a=cls.from_sympy(expr.args[0]),
                b=cls.from_sympy(expr.args[1]),
            )
        elif isinstance(expr, sp.Integer):
            return ExprBinaryTree(
                root=root, 
                a=None, 
                b=None,
            )
        else:
            raise NotImplementedError(f'Unsupported expression type {type(expr)}')
