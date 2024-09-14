import torch
import torch.nn as nn

from src.model.tree import ExprNode, INT_NE_TYPE, INT_PO_TYPE

class SympleEmbedding(nn.Module):
    def __init__(
        self,
        *embedargs,
        int_po_type: int = INT_PO_TYPE,
        int_ne_type: int = INT_NE_TYPE,
        **embedkwargs,
    ):
        super(SympleEmbedding, self).__init__()
        self.int_po_type = int_po_type
        self.int_ne_type = int_ne_type
        self.embedding = nn.Embedding(*embedargs, **embedkwargs)

    def forward(self, input: ExprNode) -> ExprNode:
        # t = input.to_tensor()
        input.embedding = self.embedding(torch.tensor(input.type))
        if input.type in (self.int_ne_type, self.int_po_type):
            input.embedding[-1] = input.arg
        input.embedding = input.embedding[None, :]
        input.a = self(input.a) if input.a is not None else None
        input.b = self(input.b) if input.b is not None else None
        return input

