from candle.utils import load_safetensors
from candle.models.bert import BertModel, Config
import json 
from dataclasses import fields
from transformers import BertTokenizer
from candle import Tensor

if __name__ == "__main__":
    model_name = "intfloat/e5-small-v2"
    file = r"C:\Users\lkreu\Downloads\e5_small.safetensors"
    config_file = r"C:\Users\lkreu\Downloads\e5_small.json"

    tensors = load_safetensors(file)
    config = Config()
    with open(config_file, "r") as f:
        raw_config = json.load(f)
        for field in fields(config):
            if field.name in raw_config:
                setattr(config, field.name, raw_config[field.name])
        

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
    tokens = Tensor(tokenized["input_ids"])
    token_type_ids = Tensor(tokenized["token_type_ids"])
    embeddings = model.forward(tokens, token_type_ids)
    # Apply average pooling
    (_n_sentence, n_tokens, _hidden_size) = embeddings.shape
    embeddings = (embeddings.sum_keepdim(1) / float(n_tokens))
    print(f"pooled embeddings: {embeddings}")