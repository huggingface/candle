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
    assert (t == 42.0).equal(Tensor(1).to_dtype(candle.u8))
    assert (t == 43.0).equal(Tensor(0).to_dtype(candle.u8))

    assert (t != 42.0).equal(Tensor(0).to_dtype(candle.u8))
    assert (t != 43.0).equal(Tensor(1).to_dtype(candle.u8))

    assert (t > 41.0).equal(Tensor(1).to_dtype(candle.u8))
    assert (t > 42.0).equal(Tensor(0).to_dtype(candle.u8))

    assert (t >= 42.0).equal(Tensor(1).to_dtype(candle.u8))
    assert (t >= 43.0).equal(Tensor(0).to_dtype(candle.u8))

    assert (t < 43.0).equal(Tensor(1).to_dtype(candle.u8))
    assert (t < 42.0).equal(Tensor(0).to_dtype(candle.u8))

    assert (t <= 42.0).equal(Tensor(1).to_dtype(candle.u8))
    assert (t <= 41.0).equal(Tensor(0).to_dtype(candle.u8))


def test_tensor_supports_equality_opperations_with_tensors():
    t = Tensor(42.0)
    same = Tensor(42.0)
    other = Tensor(43.0)

    assert (t == same).equal(Tensor(1).to_dtype(candle.u8))
    assert (t == other).equal(Tensor(0).to_dtype(candle.u8))

    assert (t != same).equal(Tensor(0).to_dtype(candle.u8))
    assert (t != other).equal(Tensor(1).to_dtype(candle.u8))

    assert (t > same).equal(Tensor(0).to_dtype(candle.u8))
    assert (t > other).equal(Tensor(0).to_dtype(candle.u8))

    assert (t >= same).equal(Tensor(1).to_dtype(candle.u8))
    assert (t >= other).equal(Tensor(0).to_dtype(candle.u8))

    assert (t < same).equal(Tensor(0).to_dtype(candle.u8))
    assert (t < other).equal(Tensor(1).to_dtype(candle.u8))

    assert (t <= same).equal(Tensor(1).to_dtype(candle.u8))
    assert (t <= other).equal(Tensor(1).to_dtype(candle.u8))


def test_tensor_equality_opperations_can_broadcast():
    # Create a decoder attention mask as a test case
    # e.g.
    # [[1,0,0]
    #  [1,1,0]
    #  [1,1,1]]
    mask_cond = candle.Tensor([0, 1, 2])
    mask = mask_cond < (mask_cond + 1).reshape((3, 1))
    assert mask.shape == (3, 3)
    assert mask.equal(Tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]).to_dtype(candle.u8))


def test_tensor_can_be_hashed():
    t = Tensor(42.0)
    other = Tensor(42.0)
    # Hash should represent the a unique tensor
    assert hash(t) != hash(other)
    assert hash(t) == hash(t)
