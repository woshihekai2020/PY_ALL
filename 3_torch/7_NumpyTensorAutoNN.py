

#1: numpy part, use CPU
'''
import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range( 500 ):

    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred - y).sum()
    print( t, loss )

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot( grad_y_pred )
    grad_h_relu = grad_y_pred.dot( w2.T )
    grad_h = grad_h_relu.copy()
    grad_h[ h < 0 ] = 0
    grad_w1 = x.T.dot( grad_h )

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
'''''

#2:GPU model
'''
import torch

dtype = torch.float
#device = torch.device( 'cpu' )
device = torch.device("cuda:0")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn( N, D_in, device= device, dtype= dtype )
y = torch.randn( N, D_out, device= device, dtype= dtype )

w1 = torch.randn( D_in, H, device= device, dtype= dtype, requires_grad= True )
w2 = torch.randn( H, D_out,device= device, dtype= dtype, requires_grad= True )

learning_rate = 1e-6
for t in range( 500 ):
    y_pred = x.mm(w1).clamp(min= 0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -=learning_rate * w1.grad
        w2 -=learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
'''

#3:Auto Grad
'''
import torch
class MyReLU( torch.autograd.Function ):

    def forward(ctx, x):
        ctx.save_for_backward( x )
        return x.clamp( min = 0 )

    def backward(ctx, grad_output ):
        x, =ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn( N, D_in, device= device )
y = torch.randn( N, D_out, device= device )

w1 = torch.randn( D_in, H, device= device, requires_grad= True )
w2 = torch.randn( H, D_out, device= device, requires_grad= True )

learning_rate = 1e-6
for t in range( 500 ):
    y_pred = MyReLU.apply( x.mm(w1) ).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print( t, loss.item() )

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
'''

#4:TensorFlow:静态图(not full) compare with pytorch
'''
import tensorflow as ts
import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10
x = tf.placeholder( tf.float32, shape= (None, D_in))
y = tf.placeholder( tf.float32, shape= (None, D_out))

w1 = tf.Variable( tf.random_normal((D_in, H)))
'''

#5:nn module
'''
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn( N, D_in )
y = torch.randn( N, D_out )

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

loss_fn = torch.nn.MSELoss( reduction= 'sum' )

learning_rate = 1e-4
for t in range( 500 ):
    y_pred = model( x )

    loss = loss_fn( y_pred, y )
    print( t, loss.item() )

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
'''

#6:optim
'''
import torch
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU( ),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss( reduction= 'sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam( model.parameters(), lr= learning_rate )

for t in range( 500 ):
    y_pred = model( x )

    loss = loss_fn( y_pred, y )
    print( t, loss.item() )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
'''

#7:self define,nn module
'''
import torch
class TwoLayerNet( torch.nn.Module ):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min= 0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn( N, D_in )
y = torch.randn( N, D_out )

model = TwoLayerNet( D_in, H, D_out )

loss_fn = torch.nn.MSELoss( reduction= 'sum' )
optimizer = torch.optim.SGD( model.parameters(), lr= 1e-4 )
for t in range( 500 ):
    y_pred = model( x )

    loss = loss_fn( y_pred, y )
    print( t, loss.item() )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''

#8:pytorch control stream, weight share
import random
import torch

class DynamicNet( torch.nn.Module ):
    def __init__(self, D_in, H, D_out ):
        super( DynamicNet, self ).__init__()
        self.input_linear = torch.nn.Linear( D_in, H )
        self.middle_linear = torch.nn.Linear( H, H )
        self.output_linear = torch.nn.Linear( H, D_out )

    def forward(self, x):
        h_relu = self.input_linear(x).clamp( min= 0 )
        for _ in range( random.randint(0, 3) ):
            h_relu = self.middle_linear( h_relu ).clamp( min= 0 )
        y_pred = self.output_linear( h_relu )
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn( N, D_in )
y = torch.randn( N, D_out )

model = DynamicNet( D_in, H, D_out )

criterion = torch.nn.MSELoss( reduction= 'sum' )
optimizer = torch.optim.SGD( model.parameters(), lr= 1e-4, momentum= 0.9 )
for t in range( 500 ):
    y_pred = model( x )

    loss = criterion( y_pred, y )
    print( t, loss.item() )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()









