
import torch.nn as nn
from torch import Tensor


class symple_embedding(nn.Embedding):
    def __init__(self, pos_integer_token: int, neg_integer_token: int,  *embed_args, **embed_kwargs):
        """

        :param pos_integer_token: token to encode positive integers
        :param neg_integer_token: token to encode negative integers
        :param embed_args: The arguments for nn.Embedding:
            :param vocab_size
            :param embedding_dimension
        :param embed_kwargs: The keyword args for nn.Embedding.

        """
        assert pos_integer_token in range(embed_args[0]) and neg_integer_token in range(embed_args[0]), "The special tokens are not in the vocabulary."
        super(symple_embedding, self).__init__(*embed_args, **embed_kwargs)
        self.pos_integer_token = pos_integer_token
        self.neg_integer_token = neg_integer_token

    def forward(self, input: Tensor, magnitude: Tensor) -> Tensor:
        """

        :param input: tokens
        :param magnitude: should be "0" for most tokens but equal to the magnitude for tokens representing positive integers, or negative integers
        :return:
        """
        y = super(symple_embedding,self).forward(input)
        y[input == self.pos_integer_token, -1] = magnitude[input == self.pos_integer_token].float()
        y[input == self.neg_integer_token, -1] = magnitude[input == self.neg_integer_token].float()
        return y
