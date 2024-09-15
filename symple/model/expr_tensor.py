from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from symple.expr import ACTIONS, ExprNode


@dataclass
class ExprTensor:
    # (L, 2)
    nodes: Tensor
    # (L, 2)
    children: Tensor
    # (L, n_actions)
    valid_actions: Tensor
    # (L)
    sorted_nodes: List[ExprNode]
    
    @classmethod
    def from_node(cls, node: ExprNode) -> "ExprTensor":
        sorted_nodes = node.topological_sort()
        L = len(sorted_nodes)
        
        nodes = torch.zeros(L, 2, dtype=torch.float32)
        children = torch.full((L, 2), fill_value=L, dtype=torch.int64)
        valid_actions = torch.zeros(L, len(ACTIONS), dtype=torch.bool)
        
        for i, node in enumerate(sorted_nodes):
            nodes[i, 0] = node.type
            nodes[i, 1] = node.arg
            if node.a is not None:
                children[i, 0] = sorted_nodes.index(node.a)
            if node.b is not None:
                children[i, 1] = sorted_nodes.index(node.b)
            for a, is_valid in enumerate(ACTIONS.values()):
                valid_actions[i, a] = is_valid(node)

        return ExprTensor(
            nodes=nodes, 
            children=children, 
            valid_actions=valid_actions,
            sorted_nodes=sorted_nodes
        )