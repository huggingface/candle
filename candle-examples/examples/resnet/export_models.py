# This script exports pre-trained model weights in the safetensors format.
import numpy as np
import torch
import torchvision
from safetensors import torch as stt

m = torchvision.models.resnet50(pretrained=True)
stt.save_file(m.state_dict(), 'resnet50.safetensors')
m = torchvision.models.resnet101(pretrained=True)
stt.save_file(m.state_dict(), 'resnet101.safetensors')
m = torchvision.models.resnet152(pretrained=True)
stt.save_file(m.state_dict(), 'resnet152.safetensors')
