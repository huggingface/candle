import candle
from candle import Tensor
from candle.utils import cuda_is_available
from candle.testing import assert_equal
import pytest


def test_tensor_can_be_constructed():
    t = Tensor(42.0)
    assert t.values() == 42.0


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
    assert t[...].values() == t.values()


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


def assert_bool(t: Tensor, expected: bool):
    assert t.shape == ()
    assert str(t.dtype) == str(candle.u8)
    assert bool(t.values()) == expected


def test_tensor_supports_equality_operations_with_scalars():
    t = Tensor(42.0)

    assert_bool(t == 42.0, True)
    assert_bool(t == 43.0, False)

    assert_bool(t != 42.0, False)
    assert_bool(t != 43.0, True)

    assert_bool(t > 41.0, True)
    assert_bool(t > 42.0, False)

    assert_bool(t >= 41.0, True)
    assert_bool(t >= 42.0, True)

    assert_bool(t < 43.0, True)
    assert_bool(t < 42.0, False)

    assert_bool(t <= 43.0, True)
    assert_bool(t <= 42.0, True)


def test_tensor_supports_equality_operations_with_tensors():
    t = Tensor(42.0)
    same = Tensor(42.0)
    other = Tensor(43.0)

    assert_bool(t == same, True)
    assert_bool(t == other, False)

    assert_bool(t != same, False)
    assert_bool(t != other, True)

    assert_bool(t > same, False)
    assert_bool(t > other, False)

    assert_bool(t >= same, True)
    assert_bool(t >= other, False)

    assert_bool(t < same, False)
    assert_bool(t < other, True)

    assert_bool(t <= same, True)
    assert_bool(t <= other, True)


def test_tensor_equality_operations_can_broadcast():
    # Create a decoder attention mask as a test case
    # e.g.
    # [[1,0,0]
    #  [1,1,0]
    #  [1,1,1]]
    mask_cond = candle.Tensor([0, 1, 2])
    mask = mask_cond < (mask_cond + 1).reshape((3, 1))
    assert mask.shape == (3, 3)
    assert_equal(mask, Tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).to_dtype(candle.u8))


def test_tensor_can_be_hashed():
    t = Tensor(42.0)
    other = Tensor(42.0)
    # Hash should represent a unique tensor
    assert hash(t) != hash(other)
    assert hash(t) == hash(t)


def test_tensor_can_be_expanded_with_none():
    t = candle.rand((12, 12))

    b = t[None]
    assert b.shape == (1, 12, 12)
    c = t[:, None, None, :]
    assert c.shape == (12, 1, 1, 12)
    d = t[None, :, None, :]
    assert d.shape == (1, 12, 1, 12)
    e = t[None, None, :, :]
    assert e.shape == (1, 1, 12, 12)
    f = t[:, :, None]
    assert f.shape == (12, 12, 1)


def test_tensor_can_be_index_via_tensor():
    t = candle.Tensor([[1, 2, 1, 2], [3, 4, 3, 4], [5, 6, 5, 6]])
    indexed = t[candle.Tensor([0, 2])]
    assert indexed.shape == (2, 4)
    assert indexed.values() == [[1, 2, 1, 2], [5, 6, 5, 6]]

    indexed = t[:, candle.Tensor([0, 2])]
    assert indexed.shape == (3, 2)
    assert indexed.values() == [[1, 1], [3, 3], [5, 5]]


def test_tensor_can_be_index_via_list():
    t = candle.Tensor([[1, 2, 1, 2], [3, 4, 3, 4], [5, 6, 5, 6]])
    indexed = t[[0, 2]]
    assert indexed.shape == (2, 4)
    assert indexed.values() == [[1, 2, 1, 2], [5, 6, 5, 6]]

    indexed = t[:, [0, 2]]
    assert indexed.shape == (3, 2)
    assert indexed.values() == [[1, 1], [3, 3], [5, 5]]


def test_tensor_can_be_cast_via_to():
    t = Tensor(42.0)
    assert str(t.dtype) == str(candle.f32)
    t_new_args = t.to(candle.f64)
    assert str(t_new_args.dtype) == str(candle.f64)
    t_new_kwargs = t.to(dtype=candle.f64)
    assert str(t_new_kwargs.dtype) == str(candle.f64)
    pytest.raises(TypeError, lambda: t.to("not a dtype"))
    pytest.raises(TypeError, lambda: t.to(dtype="not a dtype"))
    pytest.raises(TypeError, lambda: t.to(candle.f64, "not a dtype"))
    pytest.raises(TypeError, lambda: t.to())
    pytest.raises(ValueError, lambda: t.to(candle.f16, dtype=candle.f64))
    pytest.raises(ValueError, lambda: t.to(candle.f16, candle.f16))

    other = Tensor(42.0).to(candle.f64)
    t_new_other_args = t.to(other)
    assert str(t_new_other_args.dtype) == str(candle.f64)
    t_new_other_kwargs = t.to(other=other)
    assert str(t_new_other_kwargs.dtype) == str(candle.f64)


@pytest.mark.skipif(not cuda_is_available(), reason="CUDA is not available")
def test_tensor_can_be_moved_via_to():
    t = Tensor(42.0)
    assert t.device == "cpu"
    t_new_args = t.to("cuda")
    assert t_new_args.device == "cuda"
    t_new_kwargs = t.to(device="cuda")
    assert t_new_kwargs.device == "cuda"
    pytest.raises(TypeError, lambda: t.to("not a device"))
    pytest.raises(TypeError, lambda: t.to(device="not a device"))
    pytest.raises(TypeError, lambda: t.to("cuda", "not a device"))
    pytest.raises(TypeError, lambda: t.to())
    pytest.raises(ValueError, lambda: t.to("cuda", device="cpu"))
    pytest.raises(ValueError, lambda: t.to("cuda", "cuda"))

    other = Tensor(42.0).to("cuda")
    t_new_other_args = t.to(other)
    assert t_new_other_args.device == "cuda"
    t_new_other_kwargs = t.to(other=other)
    assert t_new_other_kwargs.device == "cuda"


@pytest.mark.skipif(not cuda_is_available(), reason="CUDA is not available")
def test_tensor_can_be_moved_and_cast_via_to():
    t = Tensor(42.0)
    assert t.device == "cpu"
    assert str(t.dtype) == str(candle.f32)
    t_new_args = t.to("cuda", candle.f64)
    assert t_new_args.device == "cuda"
    assert str(t_new_args.dtype) == str(candle.f64)
    t_new_kwargs = t.to(device="cuda", dtype=candle.f64)
    assert t_new_kwargs.device == "cuda"
    assert str(t_new_kwargs.dtype) == str(candle.f64)

    other = Tensor(42.0).to("cuda").to(candle.f64)
    t_new_other_args = t.to(other)
    assert t_new_other_args.device == "cuda"
    assert str(t_new_other_args.dtype) == str(candle.f64)
    t_new_other_kwargs = t.to(other=other)
    assert t_new_other_kwargs.device == "cuda"
    assert str(t_new_other_kwargs.dtype) == str(candle.f64)


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
    assert c.values() == c_native.values()
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
    assert c.values() == c_native.values()
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
    assert c.values() == c_native.values()
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
    assert c.values() == c_native.values()
    with pytest.raises(ValueError):
        d = candle.rand((3, 4, 5))
        e = candle.rand((4, 6))
        f = d / e
