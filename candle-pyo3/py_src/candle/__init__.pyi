# Generated content DO NOT EDIT
from typing import List, Optional, Tuple, Union

@staticmethod
def cat(tensors:List[Tensor], dim:int ):
    """
    Concatenate the tensors across one axis.
    """
    pass


@staticmethod
def load_ggml(path):
    """
    
    """
    pass


@staticmethod
def load_gguf(path):
    """
    
    """
    pass


@staticmethod
def load_safetensors(path):
    """
    
    """
    pass


@staticmethod
def ones(shape, *, dtype=None, device=None):
    """
    
    """
    pass


@staticmethod
def rand(shape, *, device=None):
    """
    
    """
    pass


@staticmethod
def randn(shape, *, device=None):
    """
    
    """
    pass


@staticmethod
def save_safetensors(path, tensors):
    """
    
    """
    pass


@staticmethod
def stack(tensors, dim):
    """
    
    """
    pass


@staticmethod
def tensor(vs):
    """
    
    """
    pass


@staticmethod
def zeros(shape, *, dtype=None, device=None):
    """
    
    """
    pass


class DType:
    pass


class QTensor:
    pass


class Tensor:
    def __init__(data):
        pass

    @property
    def shape(self):
        """
        Gets the tensor shape as a Python tuple.
        """
        pass


    def values($self):
        """
        Gets the tensor data as a Python value/array/array of array/...
        """
        pass




