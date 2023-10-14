import candle
from candle import Tensor
from candle.utils import cuda_is_available
from candle.functional import arange
import pytest


def test_tensor_can_be_constructed():
    t = Tensor(42.0)
    assert t.values() == 42.0


def test_tensor_can_be_added():
    t = Tensor(42.0)
    result = t + t
    assert result.values() == 84.0
    result = t + 2.0
    assert result.values() == 44.0
    a = candle.rand((3, 1, 4))
    b = candle.rand((2, 1))
    c_native = a.broadcast_add(b)
    c = a + b
    assert c.shape == (3, 2, 4)
    assert c.equal(c_native)
    with pytest.raises(ValueError):
        d = candle.rand((3, 4, 5))
        e = candle.rand((4, 6))
        f = d + e


def test_tensor_can_be_subtracted():
    t = Tensor(42.0)
    result = t - t
    assert result.values() == 0
    result = t - 2.0
    assert result.values() == 40.0
    a = candle.rand((3, 1, 4))
    b = candle.rand((2, 1))
    c_native = a.broadcast_sub(b)
    c = a - b
    assert c.shape == (3, 2, 4)
    assert c.equal(c_native)
    with pytest.raises(ValueError):
        d = candle.rand((3, 4, 5))
        e = candle.rand((4, 6))
        f = d - e


def test_tensor_can_be_multiplied():
    t = Tensor(42.0)
    result = t * t
    assert result.values() == 1764.0
    result = t * 2.0
    assert result.values() == 84.0
    a = candle.rand((3, 1, 4))
    b = candle.rand((2, 1))
    c_native = a.broadcast_mul(b)
    c = a * b
    assert c.shape == (3, 2, 4)
    assert c.equal(c_native)
    with pytest.raises(ValueError):
        d = candle.rand((3, 4, 5))
        e = candle.rand((4, 6))
        f = d * e


def test_tensor_can_be_divided():
    t = Tensor(42.0)
    result = t / t
    assert result.values() == 1.0
    result = t / 2.0
    assert result.values() == 21.0
    a = candle.rand((3, 1, 4))
    b = candle.rand((2, 1))
    c_native = a.broadcast_div(b)
    c = a / b
    assert c.shape == (3, 2, 4)
    assert c.equal(c_native)
    with pytest.raises(ValueError):
        d = candle.rand((3, 4, 5))
        e = candle.rand((4, 6))
        f = d / e


def test_tensor_can_be_constructed_from_list():
    t = Tensor([3.0, 1, 4, 1, 5, 9, 2, 6])
    assert t.values() == [3.0, 1, 4, 1, 5, 9, 2, 6]


def test_tensor_can_be_constructed_from_list_of_lists():
    t = Tensor([[3.0, 1, 4, 1], [5, 9, 2, 6]])
    assert t.values() == [[3.0, 1, 4, 1], [5, 9, 2, 6]]


def test_tensor_can_be_quantized():
    t = candle.randn((16, 256))
    for format in [
        "q4_0",
        "q4_1",
        "q5_0",
        "q5_1",
        "q8_0",
        "q2k",
        "q3k",
        "q4k",
        "q5k",
        "q8k",
    ]:
        for formatted_format in [format.upper(), format.lower()]:
            quant_t = t.quantize(formatted_format)
            assert quant_t.ggml_dtype.lower() == format.lower()
            assert quant_t.shape == t.shape


def test_tensor_can_be_meaned():
    t = Tensor([1.0, 2, 3])
    assert t.mean(-1).values() == 2.0
    assert t.mean(0).values() == 2.0


def test_tensor_can_be_cast_via_to():
    t = Tensor(42.0)
    assert str(t.dtype) == str(candle.f32)
    t_new = t.to(candle.f64)
    assert str(t_new.dtype) == str(candle.f64)


@pytest.mark.skipif(not cuda_is_available(), reason="CUDA is not available")
def test_tensor_can_be_moved_via_to():
    t = Tensor(42.0)
    assert t.device == "cpu"
    t_new = t.to("cuda")
    assert t_new.device == "cuda"


@pytest.mark.skipif(not cuda_is_available(), reason="CUDA is not available")
def test_tensor_can_be_moved_and_cast_via_to():
    t = Tensor(42.0)
    assert t.device == "cpu"
    assert str(t.dtype) == str(candle.f32)
    t_new = t.to("cuda", dtype=candle.f64)
    assert t_new.device == "cuda"
    assert str(t_new.dtype) == str(candle.f64)


def test_tensor_can_be_indexed():
    t = Tensor([[3.0, 1, 4, 1], [5, 9, 2, 6]])
    assert t[0].values() == [3.0, 1.0, 4.0, 1.0]
    assert t[1].values() == [5.0, 9.0, 2.0, 6.0]
    assert t[-1].values() == [5.0, 9.0, 2.0, 6.0]
    assert t[-2].values() == [3.0, 1.0, 4.0, 1.0]


def test_tensor_can_be_sliced():
    t = Tensor([3.0, 1, 4, 10, 5, 9, 2, 6])

    assert t[0:4].values() == [3.0, 1.0, 4.0, 10.0]
    assert t[4:8].values() == [5.0, 9.0, 2.0, 6.0]
    assert t[-4:].values() == [5.0, 9.0, 2.0, 6.0]
    assert t[:-4].values() == [3.0, 1.0, 4.0, 10.0]
    assert t[-4:-2].values() == [5.0, 9.0]


def test_tensor_can_be_sliced_2d():
    t = Tensor([[3.0, 1, 4, 1], [5, 9, 2, 6]])
    assert t[:, 0].values() == [3.0, 5]
    assert t[:, 1].values() == [1.0, 9.0]
    assert t[0, 0].values() == 3.0
    assert t[:, -1].values() == [1.0, 6.0]
    assert t[:, -4].values() == [3.0, 5]
    assert t[..., 0].values() == [3.0, 5]


def test_tensor_can_be_scliced_3d():
    t = Tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    assert t[:, :, 0].values() == [[1, 5], [9, 13]]
    assert t[:, :, 0:2].values() == [[[1, 2], [5, 6]], [[9, 10], [13, 14]]]
    assert t[:, 0, 0].values() == [1, 9]
    assert t[..., 0].values() == [[1, 5], [9, 13]]
    assert t[..., 0:2].values() == [[[1, 2], [5, 6]], [[9, 10], [13, 14]]]


def test_tensor_can_be_expanded_with_none():
    t = candle.rand((12, 12))
    c = t[:, None, None, :]
    assert c.shape == (12, 1, 1, 12)


def test_tensors_can_be_compared_with_equal():
    t = Tensor(42.0)
    other = Tensor(42.0)
    assert t.equal(other)
    t = Tensor([42.0, 42.1])
    other = Tensor([42.0, 42.0])
    assert not t.equal(other)
    t = Tensor(42.0)
    other = Tensor([42.0, 42.0])
    assert not t.equal(other)


def test_tensor_supports_equality_opperations_with_scalars():
    t = Tensor(42.0)
    assert (t == 42.0).equal(Tensor(1).to(candle.u8))
    assert (t == 43.0).equal(Tensor(0).to(candle.u8))

    assert (t != 42.0).equal(Tensor(0).to(candle.u8))
    assert (t != 43.0).equal(Tensor(1).to(candle.u8))

    assert (t > 41.0).equal(Tensor(1).to(candle.u8))
    assert (t > 42.0).equal(Tensor(0).to(candle.u8))

    assert (t >= 42.0).equal(Tensor(1).to(candle.u8))
    assert (t >= 43.0).equal(Tensor(0).to(candle.u8))

    assert (t < 43.0).equal(Tensor(1).to(candle.u8))
    assert (t < 42.0).equal(Tensor(0).to(candle.u8))

    assert (t <= 42.0).equal(Tensor(1).to(candle.u8))
    assert (t <= 41.0).equal(Tensor(0).to(candle.u8))


def test_tensor_supports_equality_opperations_with_tensors():
    t = Tensor(42.0)
    same = Tensor(42.0)
    other = Tensor(43.0)

    assert (t == same).equal(Tensor(1).to(candle.u8))
    assert (t == other).equal(Tensor(0).to(candle.u8))

    assert (t != same).equal(Tensor(0).to(candle.u8))
    assert (t != other).equal(Tensor(1).to(candle.u8))

    assert (t > same).equal(Tensor(0).to(candle.u8))
    assert (t > other).equal(Tensor(0).to(candle.u8))

    assert (t >= same).equal(Tensor(1).to(candle.u8))
    assert (t >= other).equal(Tensor(0).to(candle.u8))

    assert (t < same).equal(Tensor(0).to(candle.u8))
    assert (t < other).equal(Tensor(1).to(candle.u8))

    assert (t <= same).equal(Tensor(1).to(candle.u8))
    assert (t <= other).equal(Tensor(1).to(candle.u8))


def test_tensor_equality_opperations_can_broadcast():
    # Create a decoder attention mask as a test case
    # e.g.
    # [[1,0,0]
    #  [1,1,0]
    #  [1,1,1]]
    mask_cond = arange(0, 3)
    mask = mask_cond < (mask_cond + 1).view((mask_cond.size(-1), 1))
    assert mask.shape == (3, 3)
    assert mask.equal(Tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).to(candle.u8))
