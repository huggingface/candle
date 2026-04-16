from candle.utils import load_safetensors, save_gguf, load_gguf
from candle.models.bert import BertModel, Config
import json
from candle import Tensor
from tqdm import tqdm
from dataclasses import fields
import os
import time

from huggingface_hub import hf_hub_download
from transformers import BertTokenizer, AutoModel
import torch

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

    # Load the model
    model = BertModel(config)
    model.load_state_dict(tensors)

    hf_model = AutoModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

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

    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """Average the hidden states according to the attention mask"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    tokenized = tokenizer(sentences, padding=True)
    tokens = Tensor(tokenized["input_ids"])
    token_type_ids = Tensor(tokenized["token_type_ids"])
    attention_mask = Tensor(tokenized["attention_mask"])
    encoder_out, _ = model.forward(tokens, token_type_ids, attention_mask=attention_mask)

    hf_tokenized = tokenizer(sentences, padding=True, return_tensors="pt")
    hf_result = hf_model(**hf_tokenized)["last_hidden_state"]

    hf_pooled = average_pool(hf_result, hf_tokenized["attention_mask"])
    candle_pooled = average_pool(torch.tensor(encoder_out.values()), hf_tokenized["attention_mask"])

    loss = torch.nn.L1Loss()
    error = loss(hf_pooled, candle_pooled).mean().item()
    print(f"Mean error between torch-reference and candle: {error}")

    # Quantize all attention 'weights'
    quantized_tensors = {}
    for name, tensor in tqdm(tensors.items(), desc="Quantizing tensors to 5-Bit"):
        if name.endswith("weight") and ("attention" in name or "intermediate" in name or "output" in name):
            # check if the tensor is k-quantizable
            if tensor.shape[-1] % 256 == 0:
                new_tensor = tensor.quantize("q4k")
            else:
                new_tensor = tensor.quantize("q5_0")
            quantized_tensors[name] = new_tensor
        else:
            quantized_tensors[name] = tensor.quantize("q8_0")

    print(f"Saving quantized tensors")
    # Remove all None values from the config
    config_to_save = {k: v for k, v in config.__dict__.items() if v is not None}
    # Save the model
    quantized_model_file = "e5_small.gguf"
    save_gguf(quantized_model_file, quantized_tensors, config_to_save)

    file_size_mb = os.path.getsize(model_file) / 1024 / 1024
    file_size_mb_compressed = os.path.getsize(quantized_model_file) / 1024 / 1024
    print(f"Compressed model from {file_size_mb:.2f} MB to {file_size_mb_compressed:.2f} MB")
    # Load the model from the gguf
    tensors, raw_config = load_gguf(quantized_model_file)
    config = Config()
    for field in fields(config):
        if field.name in raw_config:
            setattr(config, field.name, raw_config[field.name])
    model = BertModel(config)
    # "embeddings.position_ids" is missing in the gguf as it is i64
    model.load_state_dict(tensors, strict=False)

    # Run the model again
    encoder_out_2, pooled_output_2 = model.forward(tokens, token_type_ids)
    encoder_out_2, pooled_output_2 = encoder_out_2.to_device("cpu"), pooled_output_2.to_device("cpu")

    candle_pooled_2 = average_pool(torch.tensor(encoder_out_2.values()), hf_tokenized["attention_mask"])
    error = loss(hf_pooled, candle_pooled_2).mean().item()
    print(f"Mean error between torch-reference and quantized-candle: {error}")
