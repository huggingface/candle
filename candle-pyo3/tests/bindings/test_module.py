import candle
from candle import Tensor, QTensor
from candle.nn import Module, Linear
from candle.utils import cuda_is_available

import pytest


def test_module_can_be_constructed():
    class A(Module):
        pass

    a = A()
    assert a is not None
    assert len(list(a.buffers())) == 0


def test_module_registers_tensors():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.t = Tensor(42.0)

    a = A()
    named_buffers = dict(a.named_buffers())
    assert len(named_buffers) == 1
    assert "t" in named_buffers


def test_module_registers_submodules():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(10, 20)

    a = A()
    named_modules = dict(a.named_modules())
    named_buffers = dict(a.named_buffers())
    assert len(named_buffers) == 2
    assert "linear" in named_modules
    assert "linear.weight" in named_buffers
    assert "linear.bias" in named_buffers


def test_module_can_dump_statedict():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(10, 20)
            self.t = Tensor(42.0)

    a = A()
    state_dict = a.state_dict()
    assert hasattr(state_dict, "_metadata")
    assert "t" in state_dict
    assert "linear.weight" in state_dict
    assert "linear.bias" in state_dict
    assert len(state_dict) == 3


def test_module_can_load_statedict():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(10, 20)
            self.t = Tensor(42.0)

    statedict = {
        "linear.weight": candle.ones((20, 10)),
        "linear.bias": candle.zeros((20,)),
        "t": Tensor(42.0),
    }
    a = A()
    a.load_state_dict(statedict)


def test_module_throws_on_shape_mismatch():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.t = Tensor(42.0)

    statedict = {
        "t": candle.ones((20,)),
    }
    a = A()
    with pytest.raises(RuntimeError) as excinfo:
        a.load_state_dict(statedict)
    assert "size mismatch" in str(excinfo.value)


def test_module_throws_on_missing_key():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.t = Tensor(42.0)

    statedict = {
        "not_t": Tensor(42.0),
    }

    a = A()
    with pytest.raises(RuntimeError) as excinfo:
        a.load_state_dict(statedict)
    assert 'Missing key(s) in state_dict: "t".' in str(excinfo.value)


def test_module_can_load_quantized_tensors():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.t = candle.randn((16, 256))
            self._quantizable_buffers.add("t")

    statedict = {
        "t": candle.ones((16, 256)).quantize("q4_0"),
    }
    a = A()
    a.load_state_dict(statedict)
    assert isinstance(a.t, QTensor)
    assert a.t.ggml_dtype == "Q4_0"


def test_module_dequantizes_tensors_automatically():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.t = candle.randn((16, 256))

    statedict = {
        "t": candle.ones((16, 256)).quantize("q4_0"),
    }
    a = A()
    a.load_state_dict(statedict)
    assert isinstance(a.t, Tensor)


@pytest.mark.skipif(not cuda_is_available(), reason="CUDA is not available")
def test_module_can_be_moved_to_cuda():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.t = candle.randn((16, 256))

    a = A()
    a.cuda()
    assert a.t.device == "cuda"


@pytest.mark.skipif(not cuda_is_available(), reason="CUDA is not available")
def test_module_can_be_moved_from_cuda_to_cpu():
    class A(Module):
        def __init__(self):
            super().__init__()
            self.t = candle.randn((16, 256))

    a = A()
    a.cuda()
    assert a.t.device == "cuda"
    a.cpu()
    assert a.t.device == "cpu"
