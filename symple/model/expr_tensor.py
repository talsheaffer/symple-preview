from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from symple.expr import ExprNode
from symple.expr.actions import ACTIONS


@dataclass
class ExprTensor:
    # (L, n_actions + 1)
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
        
        n_actions = len(ACTIONS)
        
        nodes = torch.zeros(L, n_actions + 1, dtype=torch.float32)
        children = torch.full((L, 2), fill_value=L, dtype=torch.int64)
        valid_actions = torch.zeros(L, n_actions, dtype=torch.bool)
        
        for i, node in enumerate(sorted_nodes):
            nodes[i, node.type] = 1
            nodes[i, n_actions] = node.arg
            if node.left is not None:
                children[i, 0] = sorted_nodes.index(node.left)
            if node.right is not None:
                children[i, 1] = sorted_nodes.index(node.right)
            for action_type, action in ACTIONS.items():
                valid_actions[i, action_type] = action.can_apply(node)

        return ExprTensor(
            nodes=nodes, 
            children=children, 
            valid_actions=valid_actions,
            sorted_nodes=sorted_nodes
        )