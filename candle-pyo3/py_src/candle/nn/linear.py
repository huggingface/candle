import math
from typing import Any

import candle
from candle import Tensor
from .module import Module

# See https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py


class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = candle.randn(128, 20)
        >>> output = m(input)
        >>> print(output.shape)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # Allow 'weight' to be quantized
        self._quantizable_buffers.add("weight")

        self.in_features = in_features
        self.out_features = out_features
        # TODO: Do actual initialization here: e.g. kaiming_uniform or xavier_uniform
        self.weight = candle.ones((out_features, in_features), **factory_kwargs)
        if bias:
            self.bias = candle.zeros((out_features,), **factory_kwargs)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        dims = x.shape
        last_dim = dims[-1]

        if isinstance(self.weight, candle.QTensor):
            if len(dims) < 3:
                matmul_result = self.weight.matmul_t(x).broadcast_add(self.bias)
            elif len(dims) == 3:
                b, n, m = dims
                output_shape = (b, n, self.out_features)
                re = x.reshape((b * n, m))
                matmul_result = self.weight.matmul_t(re).reshape((output_shape))
            else:
                raise NotImplementedError("'QTensor.matmul_t' is not implemented for more than 3 dimensions")

            if self.bias:
                return matmul_result.broadcast_add(self.bias)
        else:
            if self.weight.shape[-1] == last_dim and len(dims) < 3:
                w = self.weight.t()
            else:
                batch_size = dims[0]
                w = self.weight.broadcast_left((batch_size,)).t()

            x = x.matmul(w)
            if self.bias is not None:
                x = x.broadcast_add(self.bias)
            return x

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
