# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike
from candle.typing import _ArrayLike, Device, Scalar, Index, Shape
from candle import Tensor, DType, QTensor

class ONNXModel:
    """
    A wrapper around an ONNX model.
    """

    def __init__(self, path: str):
        pass

    @property
    def doc_string(self) -> str:
        """
        The doc string of the model.
        """
        pass

    @property
    def domain(self) -> str:
        """
        The domain of the operator set of the model.
        """
        pass

    def initializers(self) -> Dict[str, Tensor]:
        """
        Get the weights of the model.
        """
        pass

    @property
    def inputs(self) -> Optional[Dict[str, ONNXTensorDescription]]:
        """
        The inputs of the model.
        """
        pass

    @property
    def ir_version(self) -> int:
        """
        The version of the IR this model targets.
        """
        pass

    @property
    def model_version(self) -> int:
        """
        The version of the model.
        """
        pass

    @property
    def outputs(self) -> Optional[Dict[str, ONNXTensorDescription]]:
        """
        The outputs of the model.
        """
        pass

    @property
    def producer_name(self) -> str:
        """
        The producer of the model.
        """
        pass

    @property
    def producer_version(self) -> str:
        """
        The version of the producer of the model.
        """
        pass

    def run(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Run the model on the given inputs.
        """
        pass

class ONNXTensorDescription:
    """
    A wrapper around an ONNX tensor description.
    """

    @property
    def dtype(self) -> DType:
        """
        The data type of the tensor.
        """
        pass

    @property
    def shape(self) -> Tuple[Union[int, str, Any]]:
        """
        The shape of the tensor.
        """
        pass
