import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from math import inf

from src.model.tree import ExprNode

class BinaryTreeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *lstmargs, **lstmkwargs):
        super(BinaryTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.left_size = self.hidden_size//2
        self.right_size = self.hidden_size - self.left_size
        self.lstm = nn.LSTM(input_size, hidden_size, *lstmargs, **lstmkwargs)

    def forward(self, input: ExprNode, depth=inf) -> ExprNode:
        if depth < 0:
            return input
        input.a, input.b = (
            self(input.a, depth=depth - 1) if input.a is not None else None,
            self(input.b, depth=depth - 1) if input.b is not None else None,
        )
        input.ensure_parenthood()
        _, (input.hidden, input.cell) = self.lstm(
            input.arg_hot,
                (
                    torch.cat(
                        (
                            input.a.hidden[:,: self.left_size] if input.a is not None else torch.zeros((1, self.left_size)),
                            input.b.hidden[:,: self.right_size] if input.b is not None else torch.zeros((1, self.right_size)),
                        ),
                        dim=-1
                    ),
                    torch.cat(
                        (
                            input.a.cell[:,: self.left_size] if input.a is not None else torch.zeros((1, self.left_size)),
                            input.b.cell[:,: self.right_size] if input.b is not None else torch.zeros((1, self.right_size)),
                        ),
                        dim=-1
                    ),
                ),
            )
        return input


class NaryTreeLSTM(nn.Module):
    def __init__(self, N: int, input_size: int, hidden_size: int):
        """
        N-ary Tree LSTM from https://arxiv.org/pdf/1503.00075
        
        Args:
            input_size: size of each node's feature vector
            hidden_size: size of each node's hidden and cell state
            n: max number of ordered children
        
        Shape:
            Input:
                nodes: (n_nodes, input_size)
                children: (n_nodes, n_children)
            Output:
                output: (n_nodes, hidden_size)
        """
        
        super(NaryTreeLSTM, self).__init__()
        self.N = N
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_u = nn.Linear(input_size, hidden_size)
        
        self.U_i = nn.Linear(N * hidden_size, hidden_size, bias=False)
        self.U_f = nn.Linear(N * hidden_size, N * hidden_size, bias=False)
        self.U_u = nn.Linear(N * hidden_size, hidden_size, bias=False)
        self.U_o = nn.Linear(N * hidden_size, hidden_size, bias=False)

    def forward(self, input: Tensor, hidden: Tuple[Tensor, ...], cell: Tuple[Tensor, ...]) -> ExprNode:
        """
        Args:
            input: ExprNode
            hidden: Tuple of N tensors representing hidden states
            cell: Tuple of N tensors representing cell states
        """
        hidden = torch.cat(hidden, dim=-1)
        cell = torch.cat(cell, dim=-1)
        
        i = torch.sigmoid(self.W_i(input) + self.U_i(hidden))
        f = torch.sigmoid(self.W_f(input) + self.U_f(cell).view(hidden.shape[-2], self.N, self.hidden_size))
        o = torch.sigmoid(self.W_o(input) + self.U_o(hidden))
        u = torch.tanh(self.W_u(input) + self.U_u(hidden))
        
        cell = (i * u) + (f * cell.view(hidden.shape[-2], self.N, self.hidden_size)).sum(dim=-2)
        hidden = o * torch.tanh(cell)

        return hidden, cell

