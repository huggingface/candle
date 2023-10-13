from candle.models.llama import LlamaModel, Config, LlamaForCausalLM
from transformers import AutoTokenizer
import candle
from candle import utils
import json


config = Config()
model = LlamaForCausalLM(config)
state_dict = model.state_dict()

tensors = utils.load_safetensors(r"C:\Users\luk\Downloads\open_llama.safetensors")
qtensors, _ = utils.load_gguf(r"C:\Users\luk\Downloads\yarn-llama-2-7b-64k.Q3_K_S.gguf")
model.load_state_dict(qtensors)
exported = model.gguf()

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
