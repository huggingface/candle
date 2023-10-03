import candle
from candle import Tensor


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
