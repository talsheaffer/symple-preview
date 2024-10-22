from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Tuple, List
from collections import deque

import torch

from src.model.tree import ExprNode, SymbolNode, SYMBOL_VOCAB_SIZE, VOCAB_SIZE
from src.model.actions import NUM_OPS
from symple.expr.expr_node import ExprNodeType


import sympy as sp

from torch import Tensor

ACTION_MEMORY_LENGTH = 10
TELEPORT_INDEX = NUM_OPS
STATE_VECTOR_SIZE = 8*((
    NUM_OPS 
    + 1 # teleport
    + 1 # current node count (normalized)
    + 1 # total node count (normalized)
    + 1 # number of remaining checkpoints to save
)//8 + 1) # Make sure this is a multiple of 8

NUM_CHECKPOINT_STATES = 30

class SymbolManager(dict):
    def __init__(self):
        super().__init__()
        self.next_id = VOCAB_SIZE - SYMBOL_VOCAB_SIZE
        self.available = deque()

    def __getitem__(self, key):
        if key not in self:
            assert self.can_declare_symbol(), f"Cannot assign more than {SYMBOL_VOCAB_SIZE} symbols"
            if self.available:
                super().__setitem__(key, self.available.popleft())
            else:
                super().__setitem__(key, self.next_id)
                self.next_id += 1
        return super().__getitem__(key)

    def __delitem__(self, key):
        if key in self:
            self.available.append(self[key])
            super().__delitem__(key)

    def can_declare_symbol(self) -> bool:
        return len(self) < SYMBOL_VOCAB_SIZE


@dataclass
class SympleState:
    en: ExprNode = None 
    coord: tuple[int, ...] = ()
    h_glob: Tensor = None
    c_glob: Tensor = None
    symbols: SymbolManager = field(default_factory=SymbolManager)
    sub_states: Dict[str, Tuple[SymbolNode, ExprNode, Tensor, Tensor]] = field(default_factory=dict)
    primary_state: Tuple[ExprNode, Tensor, Tensor] = field(default_factory=lambda: (None, None, None))
    current_name: Optional[str] = None
    nc: Optional[int] = None
    action_record: deque = field(default_factory=lambda: deque(maxlen=ACTION_MEMORY_LENGTH))
    teleport_index: int = TELEPORT_INDEX
    checkpoint_states: Dict[str, List[Tuple[ExprNode, Tensor, Tensor]]] = field(default_factory=dict)

    def Expr_Node_from_sympy(self, expr: Union[str, sp.Expr], evaluate: bool = True) -> ExprNode:
        if isinstance(expr, str):
            expr = sp.sympify(expr, evaluate=evaluate)

        if isinstance(expr, sp.Add):
            left = self.Expr_Node_from_sympy(expr.args[0], evaluate=evaluate)
            right = self.Expr_Node_from_sympy(sp.Add(*expr.args[1:], evaluate=evaluate), evaluate=evaluate)
            return ExprNode(ExprNodeType.ADD, left=left, right=right)

        elif isinstance(expr, sp.Mul):
            left = self.Expr_Node_from_sympy(expr.args[0], evaluate=evaluate)
            right = self.Expr_Node_from_sympy(sp.Mul(*expr.args[1:], evaluate=evaluate), evaluate=evaluate)
            if left.type == ExprNodeType.INT:
                if left.arg == 1:
                    return right
                elif left.arg == -1:
                    return ExprNode(ExprNodeType.NEG, left=right)
            return ExprNode(ExprNodeType.MUL, left=left, right=right)

        elif isinstance(expr, sp.Pow):
            if expr.exp == -1:
                return ExprNode(ExprNodeType.INV, left=self.Expr_Node_from_sympy(expr.base))
            
            left = self.Expr_Node_from_sympy(expr.base, evaluate=evaluate)
            right = self.Expr_Node_from_sympy(expr.exp, evaluate=evaluate)
            return ExprNode(ExprNodeType.POW, left=left, right=right)

        elif isinstance(expr, (sp.Rational, int)):
            if isinstance(expr, (sp.Integer, int)):
                expr = ExprNode(ExprNodeType.INT, int(expr))
                if expr.arg < 0:
                    expr.arg = -expr.arg
                    return ExprNode(ExprNodeType.NEG, left=expr)
                return expr
            
            left = self.Expr_Node_from_sympy(expr.p, evaluate=evaluate)
            right = self.Expr_Node_from_sympy(expr.q, evaluate=evaluate)
            right_inv = ExprNode(ExprNodeType.INV, left=right)
            return ExprNode(ExprNodeType.MUL, left=left, right=right_inv)
        
        elif expr.is_Symbol:
            return SymbolNode(
                expr.name,
                self.symbols[expr.name]
            )
        else:
            raise ValueError(f"Unsupported expression: {expr}")

    @classmethod
    def from_sympy(cls, expr: Union[str, sp.Expr], **init_kwargs) -> "SympleState":
        state = SympleState(**init_kwargs)
        state.en = state.Expr_Node_from_sympy(expr)
        state.primary_state = (state.en, state.h_glob, state.c_glob)
        return state

    def to_sympy(self, node: Optional[ExprNode] = None) -> sp.Expr:
        if node is None:
            node = self.en
        if node.type == ExprNodeType.ADD:
            return sp.Add(self.to_sympy(node.left), self.to_sympy(node.right))
        elif node.type == ExprNodeType.NEG:
            return sp.Mul(sp.Integer(-1), self.to_sympy(node.left))
        elif node.type == ExprNodeType.MUL:
            return sp.Mul(self.to_sympy(node.left), self.to_sympy(node.right))
        elif node.type == ExprNodeType.INV:
            return sp.Pow(self.to_sympy(node.left), sp.Integer(-1))
        elif node.type == ExprNodeType.POW:
            return sp.Pow(self.to_sympy(node.left), self.to_sympy(node.right))
        elif node.type == ExprNodeType.INT:
            return sp.Integer(node.arg)
        elif node.type in self.symbols.values():
            name = [name for name, type_ in self.symbols.items() if type_ == node.type][0]
            return sp.Symbol(name)
        else:
            raise ValueError(f"Unsupported ExprNodeType: {node.type}")

    def substitute(self, node: ExprNode, new_node: ExprNode) -> None:
        self.en = self.en.substitute(node, new_node)
        for name, (n, sub_en, h, c) in self.sub_states.items():
            self.sub_states[name] = (n, sub_en.substitute(node, new_node), h, c)
        if self.current_name is not None:
            self.primary_state = (self.primary_state[0].substitute(node, new_node), *self.primary_state[1:])

    @property
    def current_node(self) -> ExprNode:
        return self.en.get_node(self.coord)
    
    def substitute_current_node(self, new_node: ExprNode) -> None:
        self.en = self.en.apply_at_coord(self.coord, lambda x: new_node)

    @property
    def n_available_checkpoints(self) -> int:
        return NUM_CHECKPOINT_STATES - sum(len(v) for v in self.checkpoint_states.values())
    
    @property
    def state_tensor(self) -> Tensor:
        """
        Returns a tensor representation of the current state.
        """

        # Encode the last action taken (one-hot encoding with vector of length 30)
        state_tensor = torch.zeros(STATE_VECTOR_SIZE, dtype=torch.float32)
        if self.action_record:
            last_action = self.action_record[-1]
            state_tensor[last_action] = 1.0

        # Calculate normalized node counts
        state_tensor[-1] = self.en.node_count() / 64.0
        state_tensor[-2] = self.current_node.node_count() / 64.0

        # Number of remaining checkpoints to save
        state_tensor[-3] = self.n_available_checkpoints / NUM_CHECKPOINT_STATES

        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

        return state_tensor
    
    def update(self) -> None:
        name = self.current_name
        if name is None:
            self.primary_state = (self.en, self.h_glob, self.c_glob)
        elif name in self.sub_states.keys():
            self.sub_states[name] = (self.sub_states[name][0], self.en, self.h_glob, self.c_glob)
        else:
            raise ValueError(f"Symbol {name} not found")

    def can_declare_symbol(self) -> bool:
        return self.symbols.can_declare_symbol()
    
    def declare_new_symbol(self, name: Optional[str] = None) -> int:
        if name is None:
            name = '_'
            while name in self.symbols:
                name += '_'
        assert name not in self.symbols, f"Symbol {name} already exists"

        current_node = self.current_node        
        type_ = self.symbols[name]
        new_node = SymbolNode(
            name,
            type_,
            hidden = current_node.hidden.clone() if current_node.hidden is not None else None,
            cell = current_node.cell.clone() if current_node.cell is not None else None
        )
        self.substitute(current_node, new_node)

        self.sub_states[name] = (
            new_node,
            current_node,
            self.h_glob.clone() if self.h_glob is not None else None,
            self.c_glob.clone() if self.c_glob is not None else None
        )

    @property
    def variable_types(self) -> Dict[int, str]:
        return {sn.type : name for name, (sn, _, _, _) in self.sub_states.items()}
    
    def can_switch_to_sub_state(self) -> bool:
        return self.current_node.type in self.variable_types
    
    def switch_to_sub_state(self, name: Optional[str] = None) -> None:
        if name is None:
            t = self.current_node.type
            name = self.variable_types.get(t, None)
            if name is None:
                raise ValueError(f"Symbol type {t} not found")
            
        assert name in self.sub_states.keys(), f"Symbol {name} not found"
        self.update()
        self.current_name = name
        _, self.en, self.h_glob, self.c_glob = self.sub_states[name]
        self.coord = ()
    
    def can_revert_to_primary_state(self) -> bool:
        return self.current_name is not None
    
    def revert_to_primary_state(self) -> None:
        self.update()
        self.en, self.h_glob, self.c_glob = self.primary_state
        self.coord = ()
        self.current_name = None
    
    def can_evaluate_symbol(self) -> bool:
        return self.current_name is not None
    
    def evaluate_symbol(self, name: Optional[str] = None) -> None:
        if name is None:
            name = self.current_name
        if name is None:
            raise ValueError("No symbol to evaluate")
        assert name in self.sub_states.keys(), f"Symbol {name} not found"
        if name == self.current_name:
            self.revert_to_primary_state()

        n, sub_en, _, _ = self.sub_states.pop(name)
        del self.symbols[name]
        if name in self.checkpoint_states.keys():
            del self.checkpoint_states[name]
        self.substitute(n, sub_en)
    
    def evaluate_all_symbols(self) -> None:
        names = list(self.sub_states.keys())
        for name in names:
            self.evaluate_symbol(name)
    
    def finish(self) -> None:
        self.evaluate_all_symbols()
        self.en.reset_tensors()
    
    def count_symbol(self, name: Optional[str] = None) -> int:
        if name is None:
            name = self.current_name
        if name is None:
            return 1
        assert name in self.sub_states.keys(), f"Symbol {name} not found"
        self.update()
        node = self.sub_states[name][1]
        count = len(self.primary_state[0].matches(node))
        for sub_name, (_, sub_en, _, _) in self.sub_states.items():
            if sub_name != name:
                sub_count = len(sub_en.matches(node))
                if sub_count > 0:
                    count += sub_count * self.count_symbol(sub_name)
        return count

    def node_count(self, name: Optional[str] = None) -> int:
        if name is None or name == 'primary':
            self.update()
            node = self.primary_state[0]
        else:
            assert name in self.sub_states.keys(), f"Symbol {name} not found"
            self.update()
            node = self.sub_states[name][1]
        
        node_types = [n.type for n in node.topological_sort()]
        variable_types = self.variable_types
        # Initialize counters
        symbol_counts = {
            key: 0 for key in variable_types.values()# if key is not name
        }
        count = 0
        for node_type in node_types:
            if node_type not in variable_types.keys():
                count += 1
            else:
                symbol_counts[variable_types[node_type]] += 1
        
        for key, value in symbol_counts.items():
            if value > 0:
                count += value * self.node_count(key)
        return count
    
    def can_save_checkpoint(self) -> bool:
        return self.n_available_checkpoints > 0
    
    def save_checkpoint(self) -> None:
        name = self.current_name if self.current_name is not None else 'primary'
        self.checkpoint_states[name] = self.checkpoint_states.get(name, []) + [
            (
                self.en.clone(),
                self.h_glob.clone() if self.h_glob is not None else None,
                self.c_glob.clone() if self.c_glob is not None else None,
                self.coord
            )
        ]
    
    def can_toggle_checkpoint(self) -> bool:
        name = self.current_name if self.current_name is not None else 'primary'
        return len(self.checkpoint_states.get(name, [])) > 0
    
    def toggle_checkpoint(self, index : Optional[int] = None) -> int:
        name = self.current_name if self.current_name is not None else 'primary'
        initial_node_count = self.node_count()
        index = index if index is not None else len(self.checkpoint_states[name])
        self.save_checkpoint()
        # Cycle the current state to the back of the list
        self.checkpoint_states[name] = list(self.checkpoint_states[name][-1:]) + list(self.checkpoint_states[name][:-1])
        # load the most recent, or indicated, checkpoint and remove it from the list
        self.en, self.h_glob, self.c_glob, self.coord = self.checkpoint_states[name].pop(index)
        current_node_count = self.node_count()
        self.update()
        return initial_node_count - current_node_count # NCR
    
    def can_revert_to_best_checkpoint(self) -> bool:
        name = self.current_name if self.current_name is not None else 'primary'
        return len(self.checkpoint_states.get(name, [])) > 0
    
    def revert_to_best_checkpoint(self) -> int:
        name = self.current_name if self.current_name is not None else 'primary'
        n_checkpoints = len(self.checkpoint_states[name])
        best_index = None
        initial_node_count = self.node_count()
        best_node_count = initial_node_count
        for i in reversed(range(n_checkpoints)):
            self.toggle_checkpoint()
            current_node_count = self.node_count()
            if current_node_count < best_node_count:
                best_index = i
                best_node_count = current_node_count
        self.toggle_checkpoint()
        if best_index is not None:
            self.toggle_checkpoint(best_index)
        return initial_node_count - best_node_count # NCR



    def __repr__(self) -> str:
        s = self.en.__repr__()
        s += f"\ncoord: {self.coord}"
        for name, (symbol_node, sub_en, _, _) in self.sub_states.items():
            s += f"\n{name}: {sub_en.__repr__()}"
        return s

