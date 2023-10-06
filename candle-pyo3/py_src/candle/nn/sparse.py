from .module import Module
from typing import Optional, Tuple, Any
from candle import Tensor
import candle


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None) -> None:
        factory_kwargs = {"device": device}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = candle.randn((num_embeddings, embedding_dim), **factory_kwargs)

    def forward(self, indexes: Tensor) -> Tensor:
        final_dims = list(indexes.shape)
        final_dims.append(self.embedding_dim)
        indexes = indexes.flatten_all()
        values = self.weight.index_select(indexes, 0)
        return values.reshape(final_dims)
