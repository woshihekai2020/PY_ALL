
numExcute=7
#1: CPU
#2: GPU
#3: Auto Grad
#4: tf similar to pytorch flow
#5: nn module
#6: optim
#7: self define nn module
#8: pytorch control stream, weight share

#1: numpy part, use CPU
if numExcute == 1 :
    import numpy as np

    #N: batch size.      D_in: input dimension
    #H: hiden dimension. D_out: output dimension
    N, D_in, H, D_out = 64, 1000, 100, 10

    #create input output data randonly
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # randonly init weight
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    learning_rate = 1e-6
    for t in range( 500 ):
        # forward spread: caculate est_y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # caculate the loss
        loss = np.square(y_pred - y).sum()
        print( t, loss )

        # backward spread. caculate grad of w1, w2 to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot( grad_y_pred )
        grad_h_relu = grad_y_pred.dot( w2.T )
        grad_h = grad_h_relu.copy()
        grad_h[ h < 0 ] = 0
        grad_w1 = x.T.dot( grad_h )

        # update weight
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

#2:GPU model
if numExcute == 2 :
    import torch

    dtype = torch.float
    #device = torch.device( 'cpu' )
    device = torch.device("cuda:0") #run GPU

    #batch size, in dimension, hiden dimension, out dimension
    N, D_in, H, D_out = 64, 1000, 100, 10

    #randanly create input and output date
    x = torch.randn( N, D_in, device= device, dtype= dtype )
    y = torch.randn( N, D_out, device= device, dtype= dtype )

    #randanly init weight
    w1 = torch.randn( D_in, H, device= device, dtype= dtype )
    w2 = torch.randn( H, D_out,device= device, dtype= dtype )

    learning_rate = 1e-6
    for t in range( 500 ):
        # forward spread
        h = x.mm( w1 )
        h_relu = h.clamp( min= 0 )
        y_pred = h_relu.mm( w2 )

        # calculate loss
        loss = (y_pred - y).pow(2).sum().item()
        print(t, loss)

        # backprop caculate grad of w1 and w2 to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm( grad_y_pred )
        grad_h_relu = grad_y_pred.mm( w2.t() )
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        #update the weight with grad loss
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

#3:Auto Grad
if numExcute == 3 :
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

        # caculate backward with autograd
        loss.backward()

        with torch.no_grad():
            # use grad descend to update weight
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # set zero after backward
            w1.grad.zero_()
            w2.grad.zero_()

#4:TensorFlow:静态图(not full) compare with pytorch
if numExcute == 4 :
    #import tensorflow as ts
    import numpy as np

    N, D_in, H, D_out = 64, 1000, 100, 10
    #x = tf.placeholder( tf.float32, shape= (None, D_in))
    #y = tf.placeholder( tf.float32, shape= (None, D_out))

    #w1 = tf.Variable( tf.random_normal((D_in, H)))

#5:nn module
if numExcute == 5 :
    import torch

    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn( N, D_in )
    y = torch.randn( N, D_out )

    #use nn module, we define series of level
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

        #set grad to zero, after backward
        model.zero_grad()

        loss.backward()

        #use grad descent to update weight
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

#6:optim
if numExcute == 6 :
    import torch
    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    #use nn to define model, and loss_function
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU( ),
        torch.nn.Linear(H, D_out),
    )
    loss_fn = torch.nn.MSELoss( reduction= 'sum')

    #use Adam to optimize model weights
    learning_rate = 1e-4
    optimizer = torch.optim.Adam( model.parameters(), lr= learning_rate )

    for t in range( 500 ):
        y_pred = model( x )

        loss = loss_fn( y_pred, y )
        print( t, loss.item() )

        optimizer.zero_grad()

        loss.backward()

        #use step() to update all params
        optimizer.step()

#7:self define,nn module
if numExcute == 7 :
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

    #init our model with our defined model
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

#8:pytorch control stream, weight share
if numExcute == 8 :
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









