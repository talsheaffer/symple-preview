import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DenseBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = None,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.dropout = dropout
        self.leaky_relu_slope = leaky_relu_slope

    def forward(self, input: Tensor) -> Tensor:
        x = F.leaky_relu(self.linear(input), negative_slope=self.leaky_relu_slope)
        if self.training:
            x = F.dropout(x, p=self.dropout)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.input = DenseBlock(self.input_size, self.hidden_size, **kwargs)
        self.layers = nn.Sequential(
            *(DenseBlock(self.hidden_size, **kwargs) for _ in range(self.n_layers))
        )
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        return self.output(self.layers(self.input(x)))

class FFN_V2(FFN):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, **kwargs):
        super().__init__(input_size, hidden_size, output_size, **kwargs)
        self.wide = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.wide(x) + super(FFN_V2, self).forward(x)
