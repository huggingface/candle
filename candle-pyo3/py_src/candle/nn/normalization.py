import candle
from candle import Tensor
from .module import Module
from typing import Union, List, Tuple, Optional, Any

_shape_t = Union[int, List[int]]
import numbers


class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`

    math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    """

    __constants__ = ["normalized_shape", "eps"]
    normalized_shape: Tuple[int, ...]
    eps: float

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.weight = candle.ones(normalized_shape, **factory_kwargs)
        if bias:
            self.bias = candle.zeros(normalized_shape, **factory_kwargs)
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        mean_x = input.sum_keepdim(2) / float(self.normalized_shape[-1])
        x = input.broadcast_sub(mean_x)
        norm_x = x.sqr().sum_keepdim(2) / float(self.normalized_shape[-1])
        x_normed = x.broadcast_div((norm_x + self.eps).sqrt())
        x = x_normed.broadcast_mul(self.weight)

        if self.bias:
            x = x.broadcast_add(self.bias)
        return x

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)
