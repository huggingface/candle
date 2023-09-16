# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike
from candle.typing import _ArrayLike, Device

class bf16(DType):
    pass

@staticmethod
def cat(tensors: List[Tensor], dim: int):
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
def ones(shape: Sequence[int], dtype: Optional[DType] = None, device: Optional[Device] = None):
    """ """
    pass

@staticmethod
def rand(shape: Sequence[int], device: Optional[Device] = None):
    """
    Creates a new tensor with random values.
    """
    pass

@staticmethod
def randn(shape: Sequence[int], device: Optional[Device] = None):
    """ """
    pass

@staticmethod
def stack(tensors: List[Tensor], dim: int):
    """
    Stack the tensors along a new axis.
    """
    pass

@staticmethod
def tensor(data: _ArrayLike):
    """
    Creates a new tensor from a Python value. The value can be a scalar or array-like object.
    """
    pass

class u32(DType):
    pass

class u8(DType):
    pass

@staticmethod
def zeros(shape: Sequence[int], dtype: Optional[DType] = None, device: Optional[Device] = None):
    """ """
    pass

class DType:
    pass

class QTensor:
    def dequantize(self):
        """ """
        pass
    @property
    def ggml_dtype(self):
        """ """
        pass
    def matmul_t(self, lhs):
        """ """
        pass
    @property
    def rank(self):
        """ """
        pass
    @property
    def shape(self):
        """ """
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
    def contiguous(self):
        """ """
        pass
    def copy(self):
        """ """
        pass
    def cos(self):
        """ """
        pass
    def detach(self):
        """ """
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
    def flatten_all(self):
        """ """
        pass
    def flatten_from(self, dim):
        """ """
        pass
    def flatten_to(self, dim):
        """ """
        pass
    def get(self, index):
        """ """
        pass
    def index_select(self, rhs, dim):
        """ """
        pass
    def is_contiguous(self):
        """ """
        pass
    def is_fortran_contiguous(self):
        """ """
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
    def mean_all(self):
        """ """
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
    def quantize(self, quantized_dtype):
        """ """
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
    def sum_all(self):
        """ """
        pass
    def sum_keepdim(self, dims):
        """ """
        pass
    def t(self):
        """ """
        pass
    def to_device(self, device):
        """ """
        pass
    def to_dtype(self, dtype):
        """ """
        pass
    def transpose(self, dim1, dim2):
        """ """
        pass
    def unsqueeze(self, dim):
        """ """
        pass
    def values(self):
        """
        Gets the tensor's data as a Python scalar or array-like object.
        """
        pass
    def where_cond(self, on_true, on_false):
        """ """
        pass
