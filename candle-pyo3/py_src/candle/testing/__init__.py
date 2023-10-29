import candle
from candle import Tensor


_UNSIGNED_DTYPES = set([str(candle.u8), str(candle.u32)])


def _assert_tensor_metadata(
    actual: Tensor,
    expected: Tensor,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
):
    if check_device:
        assert actual.device == expected.device, f"Device mismatch: {actual.device} != {expected.device}"

    if check_dtype:
        assert str(actual.dtype) == str(expected.dtype), f"Dtype mismatch: {actual.dtype} != {expected.dtype}"

    if check_layout:
        assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} != {expected.shape}"

    if check_stride:
        assert actual.stride == expected.stride, f"Stride mismatch: {actual.stride} != {expected.stride}"


def assert_equal(
    actual: Tensor,
    expected: Tensor,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
):
    """
    Asserts that two tensors are exact equals.
    """
    _assert_tensor_metadata(actual, expected, check_device, check_dtype, check_layout, check_stride)
    assert (actual - expected).abs().sum_all().values() == 0, f"Tensors mismatch: {actual} != {expected}"


def assert_almost_equal(
    actual: Tensor,
    expected: Tensor,
    rtol=1e-05,
    atol=1e-08,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
):
    """
    Asserts, that two tensors are almost equal by performing an element wise comparison of the tensors with a tolerance.

    Computes: |actual - expected| ≤ atol + rtol x |expected|
    """
    _assert_tensor_metadata(actual, expected, check_device, check_dtype, check_layout, check_stride)

    # Secure against overflow of u32 and u8 tensors
    if str(actual.dtype) in _UNSIGNED_DTYPES or str(expected.dtype) in _UNSIGNED_DTYPES:
        actual = actual.to(candle.i64)
        expected = expected.to(candle.i64)

    diff = (actual - expected).abs()

    threshold = (expected.abs().to_dtype(candle.f32) * rtol + atol).to(expected)

    assert (diff <= threshold).sum_all().values() == actual.nelement, f"Difference between tensors was to great"
