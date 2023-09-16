# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from os import PathLike
from candle.typing import _ArrayLike, Device
from candle import Tensor, DType

@staticmethod
def cuda_is_available():
    """
    Returns true if the 'cuda' backend is available.
    """
    pass

@staticmethod
def get_num_threads():
    """
    Returns the number of threads used by the candle.
    """
    pass

@staticmethod
def has_accelerate():
    """
    Returns true if candle was compiled with 'accelerate' support.
    """
    pass

@staticmethod
def has_mkl():
    """
    Returns true if candle was compiled with MKL support.
    """
    pass

@staticmethod
def load_ggml(path: Union[str, PathLike]):
    """
    Load a GGML file. Returns a tuple of three objects: a dictionary mapping tensor names to tensors,
    a dictionary mapping hyperparameter names to hyperparameter values, and a vocabulary.
    """
    pass

@staticmethod
def load_gguf(path: Union[str, PathLike]):
    """
    Loads a GGUF file. Returns a tuple of two dictionaries: the first maps tensor names to tensors,
    and the second maps metadata keys to metadata values.
    """
    pass

@staticmethod
def load_safetensors(path: Union[str, PathLike]):
    """
    Loads a safetensors file. Returns a dictionary mapping tensor names to tensors.
    """
    pass

@staticmethod
def save_safetensors(path: Union[str, PathLike], tensors: Dict[str, Tensor]):
    """
    Saves a dictionary of tensors to a safetensors file.
    """
    pass
