from typing import Tuple

import torch
from torch import Tensor, nn

from symple.model.nary_tree_lstm import NaryTreeLSTM


class SimplifierLSTM(nn.Module):
    """
    Simplifier that uses two linear classifiers to predict the action to be taken.

    Args:
        hidden_size: the hidden size of the n-ary tree lstm

    Shape:
        Input:
            nodes: (n_nodes, input_size)
            children: (n_nodes, n_children)
            valid_actions: (n_nodes, n_actions)
        Output:
            output: (2)
    """

    def __init__(
        self,
        reader_hidden_size: int,
        policy_hidden_size: int,
        policy_layers: int,
        n_actions: int,
    ):
        super(SimplifierLSTM, self).__init__()
        self.reader = NaryTreeLSTM(N=2, input_size=n_actions + 1, hidden_size=reader_hidden_size)
        self.policy_hidden = nn.LSTM(
            input_size=reader_hidden_size,
            hidden_size=policy_hidden_size,
            num_layers=policy_layers,
        )
        self.policy_output = nn.Linear(policy_hidden_size, n_actions)

    def init_state(self) -> Tuple[Tensor, Tensor]:
        return (
            torch.zeros(
                self.policy_hidden.num_layers,
                self.policy_hidden.hidden_size,
            ),
            torch.zeros(
                self.policy_hidden.num_layers,
                self.policy_hidden.hidden_size,
            ),
        )

    def forward(
        self,
        nodes: Tensor,
        children: Tensor,
        valid_actions: Tensor,
        state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hidden = self.reader(nodes, children)
        pi_hidden, new_state = self.policy_hidden(hidden, state)
        pi = self.policy_output(pi_hidden)
        pi[valid_actions.logical_not()] = -torch.inf
        pi = nn.functional.softmax(pi.flatten()).reshape(valid_actions.shape)
        return pi, new_state
