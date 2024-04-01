# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike
from candle.typing import _ArrayLike, Device, Scalar, Index, Shape
from candle import Tensor, DType, QTensor

@staticmethod
def cuda_is_available() -> bool:
    """
    Returns true if the 'cuda' backend is available.
    """
    pass

@staticmethod
def get_num_threads() -> int:
    """
    Returns the number of threads used by the candle.
    """
    pass

@staticmethod
def has_accelerate() -> bool:
    """
    Returns true if candle was compiled with 'accelerate' support.
    """
    pass

@staticmethod
def has_mkl() -> bool:
    """
    Returns true if candle was compiled with MKL support.
    """
    pass

@staticmethod
def load_ggml(path: Union[str, PathLike]) -> Tuple[Dict[str, QTensor], Dict[str, Any], List[str]]:
    """
    Load a GGML file. Returns a tuple of three objects: a dictionary mapping tensor names to tensors,
    a dictionary mapping hyperparameter names to hyperparameter values, and a vocabulary.
    """
    pass

@staticmethod
def load_gguf(path: Union[str, PathLike]) -> Tuple[Dict[str, QTensor], Dict[str, Any]]:
    """
    Loads a GGUF file. Returns a tuple of two dictionaries: the first maps tensor names to tensors,
    and the second maps metadata keys to metadata values.
    """
    pass

@staticmethod
def load_safetensors(path: Union[str, PathLike]) -> Dict[str, Tensor]:
    """
    Loads a safetensors file. Returns a dictionary mapping tensor names to tensors.
    """
    pass

@staticmethod
def save_gguf(path: Union[str, PathLike], tensors: Dict[str, QTensor], metadata: Dict[str, Any]):
    """
    Save quanitzed tensors and metadata to a GGUF file.
    """
    pass

@staticmethod
def save_safetensors(path: Union[str, PathLike], tensors: Dict[str, Tensor]) -> None:
    """
    Saves a dictionary of tensors to a safetensors file.
    """
    pass
