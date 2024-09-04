from typing import Dict, List

import sympy as sp
import torch
from torch import nn

from model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE
from src.model.tree import ExprBinaryTree, ExprNode


class BinaryTokenTreeModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(ExprNode.TYPE_LIST),
            embedding_dim=embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size * 2,
            num_layers=1,
        )
    
    def forward(self, tree: ExprBinaryTree) -> Dict[ExprNode, torch.Tensor]:
        out_map = dict()
        hc_map = dict()

        h0 = torch.zeros(
            self.lstm.hidden_size,
            dtype=DEFAULT_DTYPE,
            device=DEFAULT_DEVICE,
        )
        c0 = h0.clone()

        for token in tree.topological_sort():
            embedded_token = self.embedding(token.to_tensor())
            ah, ac = hc_map.get(token.a, (h0, c0))
            bh, bc = hc_map.get(token.b, (h0, c0))
            hc = torch.cat(ah, bh), torch.cat(ac, bc)
            out_map[token], hc_map[token] = self.lstm(embedded_token, hc)
        
        return out_map
