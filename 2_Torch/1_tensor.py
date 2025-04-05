#https://www.pytorch123.com/SecondSection/what_is_pytorch/
# 什么是 PyTorch?
from __future__ import print_function
import torch

#############################################################################################################1: 张量初始化
print("\ntensor init:张量初始化")
x = torch.empty(5, 3)
print("~~torch.empty(5, 3): \n  ", x)
x = torch.rand(5, 3)
print("~~torch.rand(5, 3): \n  ", x)
x = torch.zeros(5, 3, dtype= torch.long)
print("~~torch.zeros(5, 3): \n  ", x)

####################################################################################################2: 特定内容的张量初始化
print("\n\n 初始化特定内容张量")
x = torch.tensor([5.5, 3])
print("~~torch.tensor([5.5, 3]) \n  ", x)
x = x.new_ones(5, 3, dtype= torch.double)
print("~~x.new_ones(5, 3) : \n  ", x)
x = torch.randn_like(x, dtype= torch.float)
print("~~randn_like[create tensor base on a tensor]: \n  ", x)
print("~~x.size:获取张量的维度信息： \n  ", x.size())

##############################################################################################################3: 张量操作
print( "\n\n 张量计算")
y = torch.randn(5, 3)
print( "~~tentsor y : \n  ", y )
print( "~~x + y: tensor x add tensor y: \n  ", x + y )
print( "~~torch.add(x, y) : \n  ", torch.add(x, y) )

###########################################################################################################4: 张量深度操作
print( "\n\n 改变张量内容")
result = torch.empty(5, 3)
torch.add( x, y, out= result )
print( "~~torch.add(x, y, out= result) , result: \n  ", result )
# 任何使张量会发生变化的操作都有一个前缀 '_'
y.add_( x )
print( "~~_y.add_(x)==> y= : \n  ", y )
print( " use like numpy: x[:, i] \n  ", x[:, 1] )
print( "\n 使用 torch.view 改变张量尺寸:")
x = torch.randn(4, 4)
y = x.view( 16 )
z = x.view( -1, 8 )
print( x.size(), y.size(), z.size() )

x = torch.randn( 1 )
print( "\n x = x.item, \n x = : \n", x )
print( " x.item() = : \n", x.item() )