# Get the checkpoint from
# https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt

import torch
from safetensors.torch import save_file

data = torch.load("tiny.en.pt")
weights = {}
for k, v in data["model_state_dict"].items():
    weights[k] = v.contiguous()
    print(k, v.shape, v.dtype)
save_file(weights, "tiny.en.safetensors")
print(data["dims"])
