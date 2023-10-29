# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike
from candle.typing import _ArrayLike, Device, Scalar, Index, Shape

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
def ones(*shape: Shape, dtype: Optional[DType] = None, device: Optional[Device] = None) -> Tensor:
    """
    Creates a new tensor filled with ones.
    """
    pass

@staticmethod
def rand(*shape: Shape, device: Optional[Device] = None) -> Tensor:
    """
    Creates a new tensor with random values.
    """
    pass

@staticmethod
def randn(*shape: Shape, device: Optional[Device] = None) -> Tensor:
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
def zeros(*shape: Shape, dtype: Optional[DType] = None, device: Optional[Device] = None) -> Tensor:
    """
    Creates a new tensor filled with zeros.
    """
    pass

class DType:
    """
    A `candle` dtype.
    """

class QTensor:
    """
    A quantized tensor.
    """

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
    def matmul_t(self, lhs: Tensor) -> Tensor:
        """
        Performs a quantized matrix multiplication, with the quantized tensor as the right hand side.
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
    """
    A `candle` tensor.
    """

    def __init__(self, data: _ArrayLike):
        pass
    def __add__(self, rhs: Union[Tensor, Scalar]) -> "Tensor":
        """
        Add a scalar to a tensor or two tensors together.
        """
        pass
    def __getitem__(self, index: Union[Index, Tensor, Sequence[Index]]) -> "Tensor":
        """
        Return a slice of a tensor.
        """
        pass
    def __mul__(self, rhs: Union[Tensor, Scalar]) -> "Tensor":
        """
        Multiply a tensor by a scalar or one tensor by another.
        """
        pass
    def __radd__(self, rhs: Union[Tensor, Scalar]) -> "Tensor":
        """
        Add a scalar to a tensor or two tensors together.
        """
        pass
    def __richcmp__(self, rhs: Union[Tensor, Scalar], op) -> "Tensor":
        """
        Compare a tensor with a scalar or one tensor with another.
        """
        pass
    def __rmul__(self, rhs: Union[Tensor, Scalar]) -> "Tensor":
        """
        Multiply a tensor by a scalar or one tensor by another.
        """
        pass
    def __sub__(self, rhs: Union[Tensor, Scalar]) -> "Tensor":
        """
        Subtract a scalar from a tensor or one tensor from another.
        """
        pass
    def __truediv__(self, rhs: Union[Tensor, Scalar]) -> "Tensor":
        """
        Divide a tensor by a scalar or one tensor by another.
        """
        pass
    def argmax_keepdim(self, dim: int) -> Tensor:
        """
        Returns the indices of the maximum value(s) across the selected dimension.
        """
        pass
    def argmin_keepdim(self, dim: int) -> Tensor:
        """
        Returns the indices of the minimum value(s) across the selected dimension.
        """
        pass
    def broadcast_add(self, rhs: Tensor) -> Tensor:
        """
        Adds the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
        """
        pass
    def broadcast_as(self, *shape: Shape) -> Tensor:
        """
        Broadcasts the tensor to the given shape.
        """
        pass
    def broadcast_div(self, rhs: Tensor) -> Tensor:
        """
        Divides the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
        """
        pass
    def broadcast_left(self, *shape: Shape) -> Tensor:
        """
        Broadcasts the tensor to the given shape, adding new dimensions on the left.
        """
        pass
    def broadcast_mul(self, rhs: Tensor) -> Tensor:
        """
        Multiplies the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
        """
        pass
    def broadcast_sub(self, rhs: Tensor) -> Tensor:
        """
        Subtracts the two tensors, while broadcasting the right-hand-side tensor to match the shape of the left-hand-side tensor.
        """
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
    def cos(self) -> Tensor:
        """
        Performs the `cos` operation on the tensor.
        """
        pass
    def detach(self) -> Tensor:
        """
        Detach the tensor from the computation graph.
        """
        pass
    @property
    def device(self) -> Device:
        """
        Gets the tensor's device.
        """
        pass
    @property
    def dtype(self) -> DType:
        """
        Gets the tensor's dtype.
        """
        pass
    def exp(self) -> Tensor:
        """
        Performs the `exp` operation on the tensor.
        """
        pass
    def flatten_all(self) -> Tensor:
        """
        Flattens the tensor into a 1D tensor.
        """
        pass
    def flatten_from(self, dim: int) -> Tensor:
        """
        Flattens the tensor on the dimension indexes from `dim` (inclusive) to the last dimension.
        """
        pass
    def flatten_to(self, dim: int) -> Tensor:
        """
        Flattens the tensor on the dimension indexes from `0` to `dim` (inclusive).
        """
        pass
    def get(self, index: int) -> Tensor:
        """
        Gets the value at the specified index.
        """
        pass
    def index_select(self, rhs: Tensor, dim: int) -> Tensor:
        """
        Select values for the input tensor at the target indexes across the specified dimension.

        The `indexes` is argument is an int tensor with a single dimension.
        The output has the same number of dimension as the `self` input. The target dimension of
        the output has length the length of `indexes` and the values are taken from `self` using
        the index from `indexes`. Other dimensions have the same number of elements as the input
        tensor.
        """
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
    def log(self) -> Tensor:
        """
        Performs the `log` operation on the tensor.
        """
        pass
    def matmul(self, rhs: Tensor) -> Tensor:
        """
        Performs a matrix multiplication between the two tensors.
        """
        pass
    def max_keepdim(self, dim: int) -> Tensor:
        """
        Gathers the maximum value across the selected dimension.
        """
        pass
    def mean_all(self) -> Tensor:
        """
        Returns the mean of the tensor.
        """
        pass
    def min_keepdim(self, dim: int) -> Tensor:
        """
        Gathers the minimum value across the selected dimension.
        """
        pass
    def narrow(self, dim: int, start: int, len: int) -> Tensor:
        """
        Returns a new tensor that is a narrowed version of the input, the dimension `dim`
        ranges from `start` to `start + len`.
        """
        pass
    def powf(self, p: float) -> Tensor:
        """
        Performs the `pow` operation on the tensor with the given exponent.
        """
        pass
    def quantize(self, quantized_dtype: str) -> QTensor:
        """
        Quantize the tensor.
        """
        pass
    @property
    def rank(self) -> int:
        """
        Gets the tensor's rank.
        """
        pass
    def recip(self) -> Tensor:
        """
        Get the `recip` of the tensor.
        """
        pass
    def reshape(self, *shape: Shape) -> Tensor:
        """
        Reshapes the tensor to the given shape.
        """
        pass
    @property
    def shape(self) -> Tuple[int]:
        """
        Gets the tensor's shape.
        """
        pass
    def sin(self) -> Tensor:
        """
        Performs the `sin` operation on the tensor.
        """
        pass
    def sqr(self) -> Tensor:
        """
        Squares the tensor.
        """
        pass
    def sqrt(self) -> Tensor:
        """
        Calculates the square root of the tensor.
        """
        pass
    def squeeze(self, dim: int) -> Tensor:
        """
        Creates a new tensor with the specified dimension removed if its size was one.
        """
        pass
    @property
    def stride(self) -> Tuple[int]:
        """
        Gets the tensor's strides.
        """
        pass
    def sum_all(self) -> Tensor:
        """
        Returns the sum of the tensor.
        """
        pass
    def sum_keepdim(self, dim: Union[int, List[int]]) -> Tensor:
        """
        Returns the sum of all elements in the input tensor. The sum is performed over all the input dimensions.
        """
        pass
    def t(self) -> Tensor:
        """
        Transposes the tensor.
        """
        pass
    def to(self, *args, **kwargs) -> Tensor:
        """
        Performs Tensor dtype and/or device conversion.
        """
        pass
    def to_device(self, device: Union[str, Device]) -> Tensor:
        """
        Move the tensor to a new device.
        """
        pass
    def to_dtype(self, dtype: Union[str, DType]) -> Tensor:
        """
        Convert the tensor to a new dtype.
        """
        pass
    def to_torch(self) -> torch.Tensor:
        """
        Converts candle's tensor to pytorch's tensor
        """
        pass
    def transpose(self, dim1: int, dim2: int) -> Tensor:
        """
        Returns a tensor that is a transposed version of the input, the given dimensions are swapped.
        """
        pass
    def unsqueeze(self, dim: int) -> Tensor:
        """
        Creates a new tensor with a dimension of size one inserted at the specified position.
        """
        pass
    def values(self) -> _ArrayLike:
        """
        Gets the tensor's data as a Python scalar or array-like object.
        """
        pass
    def where_cond(self, on_true: Tensor, on_false: Tensor) -> Tensor:
        """
        Returns a tensor with the same shape as the input tensor, the values are taken from
        `on_true` if the input tensor value is not zero, and `on_false` at the positions where the
        input tensor is equal to zero.
        """
        pass
