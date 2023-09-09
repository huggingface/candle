from typing import Any, List, Tuple, Union
from enum import Enum, auto

class Device(Enum):
    """
    Backend device for a tensor.
    """
    Cpu = auto(),
    Cuda = auto(),

class DType:
    """
    The DType of a tensor.
    """
    ...

class QTensor:
    """
    Represents a quantized `candle` Tensor.
    """
    def dequantize(self) -> Tensor:
        ...

class Tensor:
    """
    Represents a internal `candle` Tensor.
    """
    def __init__(self, data: Any) -> None: 
        """
        Create a Tensor from given data.
        """

    @property
    def shape(self) -> List[int]:
        """
        Returns the shape of the Tensor.
        """

    @property
    def rank(self) -> int:
        """
        Returns the rank of the Tensor.
        """
    
    @property
    def device(self) -> Device:
        """
        Returns the device of the Tensor.
        """

    @property
    def dtype(self) -> DType:
        """
        Returns the dtype of the Tensor.
        """

    def values(self) -> Any:
        """
        Return the values of the Tensor as a python object.
        """

    def reshape(self, shape: List[int])-> Tensor:
        """
        Reshape the Tensor.
        """

    def t(self)-> Tensor:
        """
        Transpose the Tensor.
        """

    def matmul(self, other: Tensor) -> Tensor:
        """
        Matrix multiplication.
        """

    def to_dtype(self, dtype: Union[DType, str]) -> Tensor:
        """
        Convert the Tensor to a different dtype.
        """

    def quantize(self, qtype: str) -> QTensor:
        """
        Quantize the Tensor.
        """

    def __add__(self, other: Tensor) -> Tensor:
        """
        Add two Tensors.
        """

    def __sub__(self, other: Tensor) -> Tensor:
        """
        Subtract two Tensors.
        """
    
    def sqr(self) -> Tensor:
        """
        Square the Tensor.
        """

    def mean_all(self) -> Tensor:
        """
        Mean value of the Tensor.
        """


def randn(shape:Tuple[int])->Tensor:
    """
    Create a Tensor with random values.
    """