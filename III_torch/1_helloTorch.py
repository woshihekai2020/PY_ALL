import torch
x = torch.rand(5, 3, dtype=torch.float32)
print( x )
print( torch.cuda.is_available() )

#1：tensor add
y = torch.rand( 5, 3, dtype=torch.float32 )
print( y )
y.add_( x )
print("1-exp:\n", y )

#3：data slice like numpy
y = x[:, 1]
print("3-exp:\n", y, y.shape )

#4：view
x = torch.rand(2, 2, dtype=torch.float32 )
print( x )
y = x.view( 4 )
print("4-exp:\n", y, y.shape)

#5:tensor change to array
x = torch.rand(1, dtype= torch.float32 )
print( "5-exp:\n", x )
print( x.item() )
x = torch.rand( 2, 2, dtype=torch.float32 )
print( "tensor to numpy: \n:" ,x )
print( x.numpy() )
import numpy as np
x = np.ones((2, 2))
print( "numpy to tensor: \n", x )
print( torch.from_numpy(x) )

#6:light copy
x = torch.rand(2, 2, dtype= torch.float32 )
print( "6-exp:light copy\n", x )
y = x
x += 1
print( x )
print( y )

#7:deep copy
import copy
x = torch.rand(2, 2, dtype= torch.float32 )
print( "7-exp:deep copy \n",x )
y = copy.deepcopy( x )
x += 1
print( x )
print( y )

#8:gpu,cpu status
if torch.cuda.is_available():
    device = torch.device( 'cuda')
    print( device )
    x = torch.rand( 2, 2, dtype= torch.float32, device= device )
    print( "data on gpu:\n",x )
    print( "data on cpu:\n", x.cpu().numpy() )
    print( x.to('cpu').numpy() )






















