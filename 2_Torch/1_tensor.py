#https://www.pytorch123.com/SecondSection/what_is_pytorch/
from __future__ import print_function
import torch

#1: tensor init
x = torch.empty(5, 3)
print("\n torch.empty(5, 3): \n", x)
x = torch.rand(5, 3)
print("\n torch.rand(5, 3): \n", x)
x = torch.zeros(5, 3, dtype= torch.long)
print("\n torch.zeros(5, 3): \n", x)

#2: create a tensor with SP content
x = torch.tensor([5.5, 3])
print("\n create tensor with SP content: torch.tensor([5.5, 3]) \n", x)
x = x.new_ones(5, 3, dtype= torch.double)
print("\n new_ones(5, 3) : \n", x)
x = torch.randn_like(x, dtype= torch.float)
print("\n create tensor base on a tensor: \n", x)
print("\n 获取张量的维度信息： \n", x.size0())