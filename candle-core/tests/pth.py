import torch
from collections import OrderedDict

# Write a trivial tensor to a pt file
a= torch.tensor([[1,2,3,4], [5,6,7,8]])
o = OrderedDict()
o["test"] = a
torch.save(o, "test.pt")

torch.save({"model_state_dict": o}, "test_with_key.pt")
