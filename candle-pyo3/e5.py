from candle.utils import load_safetensors, save_gguf, load_gguf
from candle.models.bert import BertModel, Config
import json
from candle import Tensor
from tqdm import tqdm
from dataclasses import fields

from huggingface_hub import hf_hub_download
from transformers import BertTokenizer

if __name__ == "__main__":
    model_name = "intfloat/e5-small-v2"
    model_file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    config_file = hf_hub_download(repo_id=model_name, filename="config.json")

    tensors = load_safetensors(model_file)
    config = Config()
    with open(config_file, "r") as f:
        raw_config = json.load(f)
        for field in fields(config):
            if field.name in raw_config:
                setattr(config, field.name, raw_config[field.name])

    # Quantize all attention 'weights'
    for name, tensor in tqdm(tensors.items(), desc="Quantizing tensors"):
        if name.endswith("weight") and "attention" in name and len(tensor.shape) >= 2:
            new_tensor = tensor.quantize("q4_0")
            tensors[name] = new_tensor

    # Load the model
    model = BertModel(config)
    model.load_state_dict(tensors)

    sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ]

    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized = tokenizer(sentences, padding=True)
    tokens = Tensor(tokenized["input_ids"]).to_device(model.device())
    token_type_ids = Tensor(tokenized["token_type_ids"]).to_device(model.device())
    encoder_out, pooled_output = model.forward(tokens, token_type_ids)
    encoder_out, pooled_output = encoder_out.to_device("cpu"), pooled_output.to_device(
        "cpu"
    )
    # Apply average pooling
    print(f"pooled embeddings: {pooled_output}")

    # Save the model as a gguf
    qtensors = {}
    quantized_count = 0
    for name, tensor in model.named_buffers():
        if name == "embeddings.position_ids":
            # TODO handle i64
            continue
        if isinstance(tensor, Tensor):
            qtensors[name] = tensor.quantize(str(tensor.dtype))
        else:
            qtensors[name] = tensor
            quantized_count += 1

    print(f"Saving with  {quantized_count} quantized tensors")
    # Remove all None values from the config
    config_to_save = {k: v for k, v in config.__dict__.items() if v is not None}
    # Save the model
    save_gguf("e5_small.gguf", qtensors, config_to_save)

    # Load the model from the gguf
    tensors, raw_config = load_gguf("e5_small.gguf")
    config = Config()
    for field in fields(config):
        if field.name in raw_config:
            setattr(config, field.name, raw_config[field.name])
    model = BertModel(config)
    # "embeddings.position_ids" is missing in the gguf as it is i64
    model.load_state_dict(tensors, strict=False)

    # Run the model again
    encoder_out_2, pooled_output_2 = model.forward(tokens, token_type_ids)
    encoder_out_2, pooled_output_2 = encoder_out_2.to_device(
        "cpu"
    ), pooled_output_2.to_device("cpu")
    print(f"pooled embeddings: {pooled_output_2}")
