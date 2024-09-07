from typing import Dict, List

from numpy import inf

import sympy as sp

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model_default import DEFAULT_DEVICE, DEFAULT_DTYPE
from tree import ExprNode, INT_NE_TYPE, INT_PO_TYPE, VOCAB_SIZE
from environment import Symple, NUM_OPS
from ffn import FFN


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

    def forward(self, input: "ExprNode") -> "ExprNode":
        # t = input.to_tensor()
        input.embedding = self.embedding(torch.tensor(input.type))
        if input.type in (self.int_ne_type, self.int_po_type):
            input.embedding[-1] = input.arg
        input.embedding = input.embedding[None, :]
        input.a, input.b = (
            self(child) if child is not None else None for child in input.children
        )
        return input


class BinaryTreeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *lstmargs, **lstmkwargs):
        super(BinaryTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, 2 * hidden_size, *lstmargs, **lstmkwargs)

    def forward(self, input: "ExprNode", depth=inf) -> "ExprNode":
        if input.a == None or input.b == None or depth < 0:
            if depth < 0:
                return input
            input.hidden = torch.zeros((1, self.hidden_size))
            input.cell = torch.zeros((1, self.hidden_size))
            return input
        input.a, input.b = (
            self(input.a, depth=depth - 1),
            self(input.b, depth=depth - 1),
        )
        _, (input.hidden, input.cell) = self.lstm(
            input.embedding,
            (
                torch.cat((input.a.hidden, input.b.hidden), dim=1),
                torch.cat((input.a.cell, input.b.cell), dim=1),
            ),
        )
        input.hidden, input.cell = (
            v[:, : self.hidden_size] for v in (input.hidden, input.cell)
        )  # Truncated to hidden size. Figure out how to use the other half, or avoid computing it.
        return input


# # debugging
#
# x = sp.Symbol('x')
# expr = sp.expand((x**2-x+1)**4)
# en = ExprNode.from_sympy(expr)
# se = SympleEmbedding(VOCAB_SIZE,8)
# en = se(en)
# print(en.embedding)
# print(se.embedding.weight.grad)
# en.embedding.norm().backward()
# print(se.embedding.weight.grad.any())
#
#
# btlstm = BinaryTreeLSTM(8, 8)
# en = btlstm(en)
# print(*(n.embedding for n in en.topological_sort()))


class SympleAgent(nn.Module):
    """
    Currently only actor, no critic. Need to implement: training
    """

    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        vocab_size: int = VOCAB_SIZE,
        num_ops=NUM_OPS,
    ):
        super(SympleAgent, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_ops = num_ops
        self.embedding = SympleEmbedding(self.vocab_size, self.embedding_size)
        self.lstm = BinaryTreeLSTM(self.embedding_size, self.hidden_size)
        self.actor = FFN(self.hidden_size, self.hidden_size, self.num_ops)

    def forward(self, env: "Symple"):
        env.state = self.embedding(env.state)
        env.state = self.lstm(env.state)
        temperature = 3
        done = False
        rewards = []
        while not done:
            logits = self.actor(env.state.hidden)
            logits = logits + torch.log(env.validity_mask)

            probs = F.softmax(logits / temperature, dim=-1)
            print(probs[probs > 0].std().item())
            action = torch.multinomial(probs, 1).item()
            reward, done = env.step(action)
            rewards.append(reward)
            env.state = self.embedding(env.state)
            env.state = self.lstm(env.state)

        return rewards


# # debug agent + environment
# x = sp.Symbol('x')
# expr = sp.expand((x**2-x+1)**4)
# en = ExprNode.from_sympy(expr)
# env = Symple(en)
# agent = SympleAgent(8,8)
# rewards = agent(env)
# print(len(rewards))
# print(rewards)


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
