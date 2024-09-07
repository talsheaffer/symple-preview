import torch
import torch.nn.functional as F
from environment import NUM_OPS, Symple
from ffn import FFN
from model_default import DEFAULT_DEVICE, DEFAULT_DTYPE
from numpy import inf
from torch import nn
from tree import INT_NE_TYPE, INT_PO_TYPE, VOCAB_SIZE, ExprNode


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
        input.a = self(input.a) if input.a is not None else None
        input.b = self(input.b) if input.b is not None else None
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

    def forward(self, env: Symple):
        env.state = self.embedding(env.state)
        env.state = self.lstm(env.state)
        temperature = 3
        done = False
        rewards = []
        while not done:
            logits = self.actor(env.state.hidden)
            logits += torch.log(env.validity_mask)

            probs = F.softmax(logits / temperature, dim=-1)
            action = torch.multinomial(probs, 1).item()
            reward, done = env.step(action)
            rewards.append(reward)
            env.state = self.embedding(env.state)
            env.state = self.lstm(env.state)

        return rewards

