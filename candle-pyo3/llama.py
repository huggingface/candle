from candle.models.llama import LlamaModel, Config, LlamaForCausalLM
from transformers import AutoTokenizer
import candle
from candle import utils
import json


config = Config.from_dict(json.load(open(r"C:\Users\luk\Downloads\config.json")))
model = LlamaForCausalLM(config)
state_dict = model.state_dict()

tensors = utils.load_safetensors(r"C:\Users\luk\Downloads\open_llama.safetensors")
model.load_state_dict(tensors, strict=False)

toknizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
prompt = ["The quick brown fox jumps over the lazy dog"]
encoded = toknizer(prompt)
input_ids = candle.Tensor(encoded["input_ids"])
attention_mask = candle.Tensor(encoded["attention_mask"])
past_key_values = None

for token_idx in range(20):
    result = model.forward(input_ids, past_key_values=past_key_values, return_dict=True)
    past_key_values = result["past_key_values"]
    m = result["logits"].get(0).argmax_keepdim(-1)
    next_token = m.values()[0][0]
    print(f"Token: {next_token}", flush=True)
    new_input = input_ids.get(0).values()
    new_input.append(next_token)
    input_ids = candle.Tensor([new_input])
