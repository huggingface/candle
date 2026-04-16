import candle
from candle import Tensor, QTensor
from candle.utils import load_safetensors, save_gguf, load_gguf, save_safetensors
from pathlib import Path

TEST_DIR = Path(__file__).parent.parent / "_workdir"
TEST_DIR.mkdir(exist_ok=True)


def test_can_roundtrip_safetensors():
    tensors = {
        "a": candle.randn((16, 256)),
        "b": candle.randn((16, 16)),
    }

    file = str(TEST_DIR / "test.safetensors")
    save_safetensors(file, tensors)
    loaded_tensors = load_safetensors(file)
    assert set(tensors.keys()) == set(loaded_tensors.keys())
    for key in tensors.keys():
        assert tensors[key].values() == loaded_tensors[key].values(), "Values are not equal"
        assert tensors[key].shape == loaded_tensors[key].shape, "Shapes are not equal"
        assert str(tensors[key].dtype) == str(loaded_tensors[key].dtype), "Dtypes are not equal"


def test_can_roundtrip_gguf():
    metadata = {
        "a": 1,
        "b": "foo",
        "c": [1, 2, 3],
        "d": [[1, 2], [3, 4]],
    }

    tensors = {
        "a": candle.randn((16, 256)).quantize("q4_0"),
        "b": candle.randn((16, 16)).quantize("f32"),
    }

    file = str(TEST_DIR / "test.gguf")
    save_gguf(file, tensors, metadata)
    loaded_tensors, loaded_metadata = load_gguf(file)

    assert set(metadata.keys()) == set(loaded_metadata.keys())
    for key in metadata.keys():
        assert metadata[key] == loaded_metadata[key]

    assert set(tensors.keys()) == set(loaded_tensors.keys())
    for key in tensors.keys():
        assert tensors[key].dequantize().values() == loaded_tensors[key].dequantize().values(), "Values are not equal"
        assert tensors[key].shape == loaded_tensors[key].shape, "Shapes are not equal"
        assert str(tensors[key].ggml_dtype) == str(loaded_tensors[key].ggml_dtype), "Dtypes are not equal"
