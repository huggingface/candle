from candle import functional as F
from candle import Tensor
import candle


def test_arange():
    tensor = F.arange(0, 10)
    assert tensor.shape == (10,)
    assert str(tensor.dtype) == str(candle.f32)
    assert tensor.values() == list(range(10))


def test_rsqrt():
    tensor = F.rsqrt(Tensor([4.0, 16.0]))
    assert tensor.shape == (2,)
    assert str(tensor.dtype) == str(candle.f32)
    assert tensor.values() == [0.5, 0.25]
