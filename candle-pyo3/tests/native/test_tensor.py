import candle
from candle import Tensor
from candle.utils import cuda_is_available
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


def test_tensor_can_be_cast_via_to():
    t = Tensor(42.0)
    assert str(t.dtype) == str(candle.f32)
    t_new_args = t.to(candle.f64)
    assert str(t_new_args.dtype) == str(candle.f64)
    t_new_kwargs = t.to(dtype=candle.f64)
    assert str(t_new_kwargs.dtype) == str(candle.f64)
    pytest.raises(TypeError, lambda: t.to("not a dtype"))
    pytest.raises(TypeError, lambda: t.to(dtype="not a dtype"))


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
