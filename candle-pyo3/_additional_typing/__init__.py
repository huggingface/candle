from typing import Union, Sequence


class Tensor:
    """
    This contains the type hints for the magic methodes of the `candle.Tensor` class.
    """

    def __add__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Add a scalar to a tensor or two tensors together.
        """
        pass

    def __radd__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Add a scalar to a tensor or two tensors together.
        """
        pass

    def __sub__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Subtract a scalar from a tensor or one tensor from another.
        """
        pass

    def __truediv__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Divide a tensor by a scalar or one tensor by another.
        """
        pass

    def __mul__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Multiply a tensor by a scalar or one tensor by another.
        """
        pass

    def __rmul__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Multiply a tensor by a scalar or one tensor by another.
        """
        pass

    def __richcmp__(self, rhs: Union["Tensor", "Scalar"], op) -> "Tensor":
        """
        Compare a tensor with a scalar or one tensor with another.
        """
        pass

    def __getitem__(self, index: Union["Index", "Tensor", Sequence["Index"]]) -> "Tensor":
        """
        Return a slice of a tensor.
        """
        pass

    def __eq__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Compare a tensor with a scalar or one tensor with another.
        """
        pass

    def __ne__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Compare a tensor with a scalar or one tensor with another.
        """
        pass

    def __lt__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Compare a tensor with a scalar or one tensor with another.
        """
        pass

    def __le__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Compare a tensor with a scalar or one tensor with another.
        """
        pass

    def __gt__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Compare a tensor with a scalar or one tensor with another.
        """
        pass

    def __ge__(self, rhs: Union["Tensor", "Scalar"]) -> "Tensor":
        """
        Compare a tensor with a scalar or one tensor with another.
        """
        pass
