from candle import Tensor
from candle import rand
import pytest


def test_absolute_shapes_are_valid():
    a = rand((10, 20))
    assert a.shape == (10, 20)

    b = rand(10, 20)
    assert b.shape == (10, 20)
    pytest.raises(OverflowError, lambda: rand((10, 20, -1)))
    pytest.raises(OverflowError, lambda: rand(-1, 20))
    pytest.raises(TypeError, lambda: rand("foo", True))


def test_relative_shapes_are_valid():
    a = rand(10, 20)
    a = a.reshape((1, -1))
    assert a.shape == (1, 200)

    b = rand(10, 20)
    b = b.reshape(-1, 1)
    assert b.shape == (200, 1)

    c = rand(10, 20)
    pytest.raises(TypeError, lambda: c.reshape(1, "foo"))
    pytest.raises(ValueError, lambda: c.reshape(1, -2))
    pytest.raises(ValueError, lambda: c.reshape((-2, 1)))
    pytest.raises(ValueError, lambda: c.reshape((0, 1)))
    pytest.raises(ValueError, lambda: c.reshape((1, -1, -1)))
