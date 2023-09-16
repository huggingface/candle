# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike
from candle.typing import _ArrayLike, Device

class bf16(DType):
    pass

@staticmethod
def cat(tensors: List[Tensor], dim: int) -> Tensor:
    """
    Concatenate the tensors across one axis.
    """
    pass

class f16(DType):
    pass

class f32(DType):
    pass

class f64(DType):
    pass

class i64(DType):
    pass

@staticmethod
def ones(shape: Sequence[int], dtype: Optional[DType] = None, device: Optional[Device] = None) -> Tensor:
    """
    Creates a new tensor filled with ones.
    """
    pass

@staticmethod
def rand(shape: Sequence[int], device: Optional[Device] = None) -> Tensor:
    """
    Creates a new tensor with random values.
    """
    pass

@staticmethod
def randn(shape: Sequence[int], device: Optional[Device] = None) -> Tensor:
    """
    Creates a new tensor with random values from a normal distribution.
    """
    pass

@staticmethod
def stack(tensors: List[Tensor], dim: int) -> Tensor:
    """
    Stack the tensors along a new axis.
    """
    pass

@staticmethod
def tensor(data: _ArrayLike) -> Tensor:
    """
    Creates a new tensor from a Python value. The value can be a scalar or array-like object.
    """
    pass

class u32(DType):
    pass

class u8(DType):
    pass

@staticmethod
def zeros(shape: Sequence[int], dtype: Optional[DType] = None, device: Optional[Device] = None) -> Tensor:
    """
    Creates a new tensor filled with zeros.
    """
    pass

class DType:
    pass

class QTensor:
    def dequantize(self) -> Tensor:
        """
        Dequantizes the tensor.
        """
        pass
    @property
    def ggml_dtype(self) -> str:
        """
        Gets the tensors quantized dtype.
        """
        pass
    def matmul_t(lhs: Tensor) -> Tensor:
        """
        Performs a quantized matrix multiplication, with the quantized tensor as the left hand side.
        """
        pass
    @property
    def rank(self) -> int:
        """
        Gets the rank of the tensor.
        """
        pass
    @property
    def shape(self) -> Tuple[int]:
        """
        Gets the shape of the tensor.
        """
        pass

class Tensor:
    def __init__(data: _ArrayLike):
        pass
    def argmax_keepdim(self, dim):
        """ """
        pass
    def argmin_keepdim(self, dim):
        """ """
        pass
    def broadcast_add(self, rhs):
        """ """
        pass
    def broadcast_as(self, shape):
        """ """
        pass
    def broadcast_div(self, rhs):
        """ """
        pass
    def broadcast_left(self, shape):
        """ """
        pass
    def broadcast_mul(self, rhs):
        """ """
        pass
    def broadcast_sub(self, rhs):
        """ """
        pass
    def contiguous(self) -> Tensor:
        """
        Makes the tensor contiguous in memory.
        """
        pass
    def copy(self) -> Tensor:
        """
        Returns a copy of the tensor.
        """
        pass
    def cos(self):
        """ """
        pass
    def detach(self) -> Tensor:
        """
        Detach the tensor from the computation graph.
        """
        pass
    @property
    def device(self):
        """ """
        pass
    @property
    def dtype(self):
        """ """
        pass
    def exp(self):
        """ """
        pass
    def flatten_all(self) -> Tensor:
        """
        Flattens the tensor into a 1D tensor.
        """
        pass
    def flatten_from(dim: int):
        """
        Flattens the tensor on the dimension indexes from `dim` (inclusive) to the last dimension.
        """
        pass
    def flatten_to(dim: int) -> Tensor:
        """
        Flattens the tensor on the dimension indexes from `0` to `dim` (inclusive).
        """
        pass
    def get(self, index):
        """ """
        pass
    def index_select(self, rhs, dim):
        """ """
        pass
    def is_contiguous(self) -> bool:
        """
        Returns true if the tensor is contiguous in C order.
        """
        pass
    def is_fortran_contiguous(self) -> bool:
        """
        Returns true if the tensor is contiguous in Fortran order.
        """
        pass
    def log(self):
        """ """
        pass
    def matmul(self, rhs):
        """ """
        pass
    def max_keepdim(self, dim):
        """ """
        pass
    def mean_all(self) -> Tensor:
        """
        Returns the mean of the tensor.
        """
        pass
    def min_keepdim(self, dim):
        """ """
        pass
    def narrow(self, dim, start, len):
        """ """
        pass
    def powf(self, p):
        """ """
        pass
    def quantize(quantized_dtype: str) -> QTensor:
        """
        Quantize the tensor.
        """
        pass
    @property
    def rank(self):
        """ """
        pass
    def recip(self):
        """ """
        pass
    def reshape(self, shape):
        """ """
        pass
    @property
    def shape(self):
        """
        Gets the tensor shape as a Python tuple.
        """
        pass
    def sin(self):
        """ """
        pass
    def sqr(self):
        """ """
        pass
    def sqrt(self):
        """ """
        pass
    def squeeze(self, dim):
        """ """
        pass
    @property
    def stride(self):
        """ """
        pass
    def sum_all(self) -> Tensor:
        """
        Returns the sum of the tensor.
        """
        pass
    def sum_keepdim(self, dims):
        """ """
        pass
    def t(self) -> Tensor:
        """
        Transposes the tensor.
        """
        pass
    def to_device(device: Union[str, Device]) -> Tensor:
        """
        Move the tensor to a new device.
        """
        pass
    def to_dtype(dtype: Union[str, DType]) -> Tensor:
        """
        Convert the tensor to a new dtype.
        """
        pass
    def transpose(self, dim1, dim2):
        """ """
        pass
    def unsqueeze(self, dim):
        """ """
        pass
    def values(self) -> _ArrayLike:
        """
        Gets the tensor's data as a Python scalar or array-like object.
        """
        pass
    def where_cond(self, on_true, on_false):
        """ """
        pass
