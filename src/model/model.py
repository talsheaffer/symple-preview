from typing import Dict, List

import sympy as sp
import torch
from torch import nn, Tensor

from model_default import DEFAULT_DEVICE, DEFAULT_DTYPE
from tree import ExprNode, INT_NE_TYPE, INT_PO_TYPE

class SympleEmbedding(nn.Embedding):
    def __init__(self, *args, int_po_type: int = INT_PO_TYPE, int_ne_type: int = INT_NE_TYPE, **kwargs):
        super(SympleEmbedding,self).__init__(*args, **kwargs)
        self.int_po_type = int_po_type
        self.int_ne_type = int_ne_type

    def forward(self, input: "ExprNode") -> "ExprNode":
        # t = input.to_tensor()
        input.embedding = super(SympleEmbedding,self).forward(torch.tensor(input.type))
        if input.type in (self.int_ne_type,self.int_po_type):
            input.embedding[-1] = input.arg
        return input

# debugging
# en = ExprNode(100,5)
# se = SympleEmbedding(400,8)
# en = se(en)
# print(en.embedding)
# print(se.weight.grad)
# en.embedding.norm().backward()
# print(se.weight.grad.any())




class BinaryTreeLSTM(nn.LSTM):
    def __init__(self, input_size: int, hidden_size: int, *args, **kwargs):
        super(BinaryTreeLSTM,self).__init__(input_size, 2*hidden_size, *args,**kwargs)

    def forward(self, input: "ExprNode") -> "ExprNode":
        if input.a == None or input.b == None:
            return input
        input.output, (input.hidden, input.cell) = super(BinaryTreeLSTM,self).forward(
            input.embedding, (
                torch.cat((input.a.hidden,input.b.hidden)),
                torch.cat((input.a.cell,input.b.cell))
            )
        )
        return input

# debug
# en.hidden = torch.zeros(8)
# en.cell = en.hidden.clone()
#
# btlstm = BinaryTreeLSTM(8, 8)
# en = btlstm(en)
# print(en)



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
    
    def forward(self, tree: ExprNode) -> Dict[ExprNode, torch.Tensor]:
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
