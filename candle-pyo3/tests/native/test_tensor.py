import candle
from candle import Tensor
from candle.utils import cuda_is_available
import pytest


def test_tensor_can_be_constructed():
    t = Tensor(42.0)
    assert t.values() == 42.0


def test_tensor_can_be_added():
    t = Tensor(42.0)
    result = t + t
    assert result.values() == 84.0


def test_tensor_can_be_subtracted():
    t = Tensor(42.0)
    result = t - t
    assert result.values() == 0


def test_tensor_can_be_multiplied():
    t = Tensor(42.0)
    result = t * t
    assert result.values() == 1764.0


def test_tensor_can_be_divided():
    t = Tensor(42.0)
    result = t / t
    assert result.values() == 1.0


def test_scalar_can_be_added_to_tensor():
    t = Tensor(42.0)
    result = t + 2.0
    assert result.values() == 44.0


def test_scalar_can_be_subtracted_from_tensor():
    t = Tensor(42.0)
    result = t - 2.0
    assert result.values() == 40.0


def test_scalar_can_be_multiplied_by_tensor():
    t = Tensor(42.0)
    result = t * 2.0
    assert result.values() == 84.0


def test_scalar_can_be_divided_by_tensor():
    t = Tensor(42.0)
    result = t / 2.0
    assert result.values() == 21.0


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
    t = Tensor([[1, 2], [3, 4]])

    assert t[None, :].values() == [[[1, 2], [3, 4]]]
