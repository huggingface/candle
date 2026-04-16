import candle
from candle import Tensor
from candle.nn import Linear


def test_linear_layer_can_be_constructed():
    linear = Linear(10, 10)
    assert linear is not None


def test_linear_layer_can_forward_a_singular_input():
    linear = Linear(384, 1536)
    input_tensor = candle.randn((8, 384))
    output = linear.forward(input_tensor)
    assert output.shape == (8, 1536)


def test_linear_layer_can_forward_a_batched_input():
    linear = Linear(384, 1536)
    input_tensor = candle.randn((16, 8, 384))
    output = linear.forward(input_tensor)
    assert output.shape == (16, 8, 1536)


def test_quantized_linear_layer_can_forward_a_singular_input():
    linear = Linear(384, 1536)
    linear.weight = linear.weight.quantize("q4_0")
    input_tensor = candle.randn((8, 384))
    output = linear.forward(input_tensor)
    assert output.shape == (8, 1536)


def test_quantized_linear_layer_can_forward_a_batched_input():
    linear = Linear(384, 1536)
    linear.weight = linear.weight.quantize("q4_0")
    input_tensor = candle.randn((16, 8, 384))
    output = linear.forward(input_tensor)
    assert output.shape == (16, 8, 1536)
