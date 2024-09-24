import torch
from torch import Tensor, nn

from symple.model.nary_tree_lstm import NaryTreeLSTM


class SimplifierBaseline(nn.Module):
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
    
    def __init__(self, reader_hidden_size: int, policy_hidden_size: int, policy_layers: int, n_actions: int):
        super(SimplifierBaseline, self).__init__()
        self.reader = NaryTreeLSTM(
            N=2, 
            input_size=2,
            hidden_size=reader_hidden_size
        )
        self.policy = nn.Sequential(
            nn.Linear(reader_hidden_size, policy_hidden_size),
            nn.ReLU(),
        )
        for _ in range(policy_layers):
            self.policy.append(nn.Linear(policy_hidden_size, policy_hidden_size))
            self.policy.append(nn.ReLU())
        self.policy.append(nn.Linear(policy_hidden_size, n_actions))

    def forward(self, nodes: Tensor, children: Tensor, valid_actions: Tensor, state) -> Tensor:
        hidden = self.reader(nodes, children)
        pi = self.policy(hidden)
        pi[valid_actions.logical_not()] = -torch.inf
        pi = nn.functional.softmax(pi.flatten()).reshape(valid_actions.shape)
        return pi, None
