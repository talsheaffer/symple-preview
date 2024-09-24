import torch
from torch import Tensor, nn


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

    def forward(self, nodes: Tensor, children: Tensor) -> Tensor:
        N, H, L = self.N, self.hidden_size, nodes.shape[0]
        
        h0 = torch.zeros(1, H)
        c0 = torch.zeros(1, H)
        
        hidden_list = [None] * (L + 1)
        cell_list = [None] * (L + 1)
        
        hidden_list[-1] = h0.clone()
        cell_list[-1] = c0.clone()

        for idx in range(L):
            in_hidden = torch.cat([hidden_list[j] for j in children[idx]], dim=1)
            in_cell = torch.cat([cell_list[j] for j in children[idx]], dim=1)
            
            i = torch.sigmoid(self.W_i(nodes[idx]) + self.U_i(in_hidden))
            f = torch.sigmoid(self.W_f(nodes[idx]) + self.U_f(in_cell).view(N, H))
            o = torch.sigmoid(self.W_o(nodes[idx]) + self.U_o(in_hidden))
            u = torch.tanh(self.W_u(nodes[idx]) + self.U_u(in_hidden))
            
            cell_list[idx] = (i * u) + (f * in_cell.view(N, H)).sum(dim=0)
            hidden_list[idx] = o * torch.tanh(cell_list[idx])

        return torch.cat(hidden_list[:-1], dim=0)
