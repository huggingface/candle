from candle.models.llama import LlamaModel, Config, LlamaForCausalLM
from transformers import AutoTokenizer
import candle

config = Config()
model = LlamaForCausalLM(config)
state_dict = model.state_dict()

toknizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

prompt = ["The quick brown fox jumps over the lazy dog"]
encoded = toknizer(prompt)
input_ids = candle.Tensor(encoded["input_ids"])
attention_mask = candle.Tensor(encoded["attention_mask"])

model.forward(input_ids, attention_mask=attention_mask)
