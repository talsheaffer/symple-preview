import sympy as sp
import torch
import torch.nn.functional as F
from numpy import inf

from typing import List, Tuple


from src.model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE
from symple.expr.expr_node import ExprNode as ExprNodeBase, ARG_NULL, ExprNodeType
SYMBOL_VOCAB_SIZE = 20
VOCAB_SIZE = len(ExprNodeType) + SYMBOL_VOCAB_SIZE


class ExprNode(ExprNodeBase):
    def __init__(
        self,
        type: int,
        arg: int = ARG_NULL,
        left: "ExprNode" = None,
        right: "ExprNode" = None,
        p: "ExprNode" = None,
        hidden: torch.Tensor = None,
        cell: torch.Tensor = None,
    ) -> None:
        super(ExprNode, self).__init__(type, arg, left, right)
        if self.left is not None:
            self.left.p = self
        if self.right is not None:
            self.right.p = self
        self.p = p
        # self.embedding = embedding# if embedding is not None else torch.nn.functional.one_hot(torch.tensor([[type]]), num_classes=VOCAB_SIZE).to(DEFAULT_DEVICE, DEFAULT_DTYPE)
        self.hidden = hidden
        self.cell = cell

    @property
    def a(self) -> "ExprNode":
        return self.left
    
    @property
    def b(self) -> "ExprNode":
        return self.right
    
    @a.setter
    def a(self, value: "ExprNode") -> None:
        self.left = value
    
    @b.setter
    def b(self, value: "ExprNode") -> None:
        self.right = value
    
    @property
    def children(self) -> list["ExprNode"]:
        return [self.left, self.right]

    def ensure_parenthood(self) -> None:
        if self.a is not None:
            self.a.p = self
        if self.b is not None:
            self.b.p = self

    def clone(self) -> "ExprNode":
        en = super().clone()
        en.ensure_parenthood()
        en.p = self.p
        en.hidden = self.hidden.clone() if self.hidden is not None else None
        en.cell = self.cell.clone() if self.cell is not None else None
        return en
    
    def node_count(self, depth: int = inf) -> int:
        if depth == 0:
            return 1
        else:
            return 1 + sum(child.node_count(depth-1) for child in self.children if child is not None)
    
    def get_node(self, coord: tuple[int, ...]) -> "ExprNode":
        if not coord:
            return self
        else:
            c = self.children[coord[0]]
            return c.get_node(coord[1:])
    
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
    
    def substitute(self, en_to_replace: "ExprNode", en_to_replace_with: "ExprNode")->"ExprNode":
        if self == en_to_replace:
            return en_to_replace_with
        else:
            self.left = self.left.substitute(en_to_replace, en_to_replace_with) if self.left is not None else None
            self.right = self.right.substitute(en_to_replace, en_to_replace_with) if self.right is not None else None
            return self
    
    def matches(self, en: "ExprNode", depth: int = inf)->List[Tuple[int, ...]]:
        if self == en:
            return [()]
        else:
            return (
                [(0,) + tup for tup in self.a.matches(en, depth-1)] if self.a is not None and depth > 0 else []
            ) + (
                [(1,) + tup for tup in self.b.matches(en, depth-1)] if self.b is not None and depth > 0 else []
            )
    
    def get_coords(self, depth: int = inf)->list[tuple[int, ...]]:
        return [()] + (
            [(0,) + tup for tup in self.a.get_coords(depth-1)] if self.a is not None and depth > 0 else []
        ) + (
            [(1,) + tup for tup in self.b.get_coords(depth-1)] if self.b is not None and depth > 0 else []
        )

    def get_coords_and_nodes(self, depth: int = inf)->list[tuple[tuple[int, ...], "ExprNode"]]:
        return [((), self)] + (
            [((0,) + c, n) for c, n in self.a.get_coords_and_nodes(depth-1)] if self.a is not None and depth > 0 else []
        ) + (
            [((1,) + c, n) for c, n in self.b.get_coords_and_nodes(depth-1)] if self.b is not None and depth > 0 else []
        )

    def reset_tensors(self)->"ExprNode":
        self.hidden = None
        self.cell = None
        if self.a is not None:
            self.a.reset_tensors()
        if self.b is not None:
            self.b.reset_tensors()
        return self

    def topological_sort(self, depth: int = inf) -> list:
        if depth == 0:
            return []
        elif self.left is None and self.right is None:
            return [self]
        elif self.right is None:
            return self.left.topological_sort(depth-1) + [self]
        elif self.left is None:
            return self.right.topological_sort(depth-1) + [self]
        else:
            return self.left.topological_sort(depth-1) + self.right.topological_sort(depth-1) + [self]
    @classmethod
    def from_expr_node_base(cls, enb: ExprNodeBase) -> "ExprNode":
        en = cls(
            enb.type,
            enb.arg,
            enb.left if (isinstance(enb.left, ExprNode) or enb.left is None) else ExprNode.from_expr_node_base(enb.left),
            enb.right if (isinstance(enb.right, ExprNode) or enb.right is None) else ExprNode.from_expr_node_base(enb.right),
        )
        en.ensure_parenthood()
        return en
    
    @classmethod
    def from_sympy(cls, expr: sp.Expr) -> "ExprNode":
        return cls.from_expr_node_base(ExprNodeBase.from_sympy(expr))
    
    def to_sympy(self, evaluate: bool = True) -> sp.Expr:
        if self.type == ExprNodeType.ADD:
            return sp.Add(self.left.to_sympy(evaluate), self.right.to_sympy(evaluate), evaluate=evaluate)
        elif self.type == ExprNodeType.NEG:
            return sp.Mul(sp.Integer(-1), self.left.to_sympy(evaluate), evaluate=evaluate)
        elif self.type == ExprNodeType.MUL:
            return sp.Mul(self.left.to_sympy(evaluate), self.right.to_sympy(evaluate), evaluate=evaluate)
        elif self.type == ExprNodeType.INV:
            return sp.Pow(self.left.to_sympy(evaluate), sp.Integer(-1), evaluate=evaluate)
        elif self.type == ExprNodeType.POW:
            return sp.Pow(self.left.to_sympy(evaluate), self.right.to_sympy(evaluate), evaluate=evaluate)
        elif self.type == ExprNodeType.INT:
            return sp.Integer(self.arg)
        elif self.type == ExprNodeType.VAR_X:
            return sp.Symbol('x')
        elif self.type == ExprNodeType.VAR_Y:
            return sp.Symbol('y')
        elif self.type == ExprNodeType.VAR_Z:
            return sp.Symbol('z')
        else:
            raise ValueError(f"Unsupported ExprNodeType: {self.type}")

    @property
    def arg_hot(self):
        return (1 if self.arg == ARG_NULL else self.arg) * F.one_hot(torch.tensor([self.type]), num_classes=VOCAB_SIZE).to(DEFAULT_DEVICE, DEFAULT_DTYPE)
    
    


class SymbolNode(ExprNode):
    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name