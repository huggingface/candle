# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike
from candle.typing import _ArrayLike, Device, Scalar, Index
from candle import Tensor, DType, QTensor

@staticmethod
def arange(start: float, end: float, step: Optional[float] = 1.0) -> Tensor:
    """
    Returns a 1-D tensor with values from the interval `[start, end)` taken with common difference `step` beginning from start.
    """
    pass

@staticmethod
def gelu(tensor: Tensor) -> Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) function to a given tensor.
    """
    pass

@staticmethod
def matmul(lhs: Tensor, rhs: Tensor) -> Tensor:
    """
    Performs a matrix multiplication.
    """
    pass

@staticmethod
def relu(tensor: Tensor) -> Tensor:
    """
    Applies the Rectified Linear Unit (ReLU) function to a given tensor.
    """
    pass

@staticmethod
def rsqrt(input: Tensor) -> Tensor:
    """
    Returns a new tensor with the reciprocal of the square-root of each of the elements of `input`.
    """
    pass

@staticmethod
def silu(tensor: Tensor) -> Tensor:
    """
    Applies the Sigmoid Linear Unit (SiLU) function to a given tensor.
    """
    pass

@staticmethod
def softmax(tensor: Tensor, dim: int) -> Tensor:
    """
    Applies the Softmax function to a given tensor.#
    """
    pass

@staticmethod
def tanh(tensor: Tensor) -> Tensor:
    """
    Applies the tanh function to a given tensor.
    """
    pass
