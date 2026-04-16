# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike
from candle.typing import _ArrayLike, Device, Scalar, Index, Shape
from candle import Tensor, DType, QTensor

@staticmethod
def avg_pool2d(tensor: Tensor, ksize: int, stride: int = 1) -> Tensor:
    """
    Applies the 2d avg-pool function to a given tensor.#
    """
    pass

@staticmethod
def gelu(tensor: Tensor) -> Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) function to a given tensor.
    """
    pass

@staticmethod
def max_pool2d(tensor: Tensor, ksize: int, stride: int = 1) -> Tensor:
    """
    Applies the 2d max-pool function to a given tensor.#
    """
    pass

@staticmethod
def relu(tensor: Tensor) -> Tensor:
    """
    Applies the Rectified Linear Unit (ReLU) function to a given tensor.
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
