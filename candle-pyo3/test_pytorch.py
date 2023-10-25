import candle
import torch

# convert from candle tensor to torch tensor
t = candle.randn((3, 512, 512))
torch_tensor = t.to_torch()
print(torch_tensor)
print(type(torch_tensor))

# convert from torch tensor to candle tensor
t = torch.randn((3, 512, 512))
candle_tensor = candle.Tensor(t)
print(candle_tensor)
print(type(candle_tensor))
