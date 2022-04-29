from __future__ import print_function
import torch
#1: .tensor([]) .empty() .rand() .zeros()
x = torch.rand(5, 3, dtype=torch.float32)
print( x )
print( torch.cuda.is_available() )

#1：tensor add
y = torch.rand( 5, 3, dtype=torch.float32 )
print( y )
y.add_( x )
print("\n1-exp:\n", y ) # can also use in one func as: torch.add( x, y, out= result )

#3：data slice like numpy
y = x[:, 1]
print("\n3-exp:\n", y, y.shape )

#4：view
x = torch.rand(2, 2, dtype=torch.float32 )
print( x )
y = x.view( 4 )
print("\n4-exp:\n", y, y.shape)

#5:tensor change to array
x = torch.rand(1, dtype= torch.float32 )
print( "\n5-exp:\n", x )
print( x.item() )
x = torch.rand( 2, 2, dtype=torch.float32 )
print( "\ntensor to numpy: \n:" ,x )
print( x.numpy() )
import numpy as np
x = np.ones((2, 2))
print( "\nnumpy to tensor: \n", x )
print( torch.from_numpy(x) )

#6:light copy
x = torch.rand(2, 2, dtype= torch.float32 )
print( "\n6-exp:light copy\n", x )
y = x
x += 1
print( x )
print( y )

#7:deep copy
import copy
x = torch.rand(2, 2, dtype= torch.float32 )
print( "\n7-exp:deep copy \n",x )
y = copy.deepcopy( x )
#x.copy_( y ) 注意 任何使张量会发生变化的操作都有一个前缀 '_'。
x += 1
print( x )
print( y )

#8:gpu,cpu status
if torch.cuda.is_available():
    device = torch.device( 'cuda')
    print( device )
    x = torch.rand( 2, 2, dtype= torch.float32, device= device )
    print( "\ndata on gpu:\n",x )
    print( "\ndata on cpu:\n", x.cpu().numpy() )
    print( x.to('cpu').numpy() )






















