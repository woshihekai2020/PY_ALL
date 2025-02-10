import numpy as np 
import torch
import random


# 1: Numpy info
def numpyInfo():  # raw is 500
    # batchSize, hide_D 
    N, D_in, H, D_out = 64, 1000, 100, 10
    # input_D,  output_D

    # random init input_data and output_data
    x = np.random.randn( N, D_in )
    y = np.random.randn( N, D_out )

    # random init input_weight and output_weight
    W1 = np.random.randn( D_in, H )
    W2 = np.random.randn( H, D_out )

    learing_rate = 1e-6
    for t in range( 50 ):
        # forward: compute y predicted
        h = x.dot( W1 )
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot( W2 )


        loss = np.square(y_pred - y).sum()
        print(t, float(loss) )

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot( grad_y_pred )
        grad_h_relu = grad_y_pred.dot( W2.T )
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_W1 = x.T.dot( grad_h )

        W1 -= learing_rate * grad_W1
        W2 -= learing_rate * grad_w2
   
# 2: Tensor Acc
def tensorAcc():  # raw is 500
    dtype = torch.float
    device = torch.device( "cuda:0")

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn( N, D_in, device= device, dtype= dtype )
    y = torch.randn( N, D_out, device= device, dtype= dtype )

    W1 = torch.randn( D_in, H, device= device, dtype= dtype )
    W2 = torch.randn( H, D_out, device= device, dtype= dtype )

    learning_rate = 1e-6
    for t in range( 50 ):
        h = x.mm( W1 )
        h_relu = h.clamp( min= 0 )
        y_pred = h_relu.mm( W2 )

        loss = (y_pred - y).pow(2).sum().item()
        print(t, dtype(loss) )

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm( grad_y_pred )
        grad_h_relu = grad_y_pred.mm( W2.t() )
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm( grad_h )

        W1 -= learning_rate * grad_w1
        W2 -= learning_rate * grad_w2   
  
# 3: Auto Grad
def autoGrad():
    dtype = torch.float
    device = torch.device( "cuda:0" )

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn( N, D_in, device= device, dtype= dtype, requires_grad= True )
    y = torch.randn( N, D_out, device= device, dtype= dtype, requires_grad= True )

    W1 = torch.randn( D_in, H, device= device, dtype= dtype, requires_grad= True )
    W2 = torch.randn( H, D_out, device= device, dtype= dtype, requires_grad= True )

    learning_rate = 1e-6
    for t in range( 500 ):
        y_pred = x.mm( W1 ).clamp( min= 0 ).mm( W2 )

        loss = (y_pred - y).pow( 2 ).sum()
        print( t,  loss.item() )

        loss.backward()

        with torch.no_grad():
            W1 -= learning_rate * W1.grad
            W2 -= learning_rate * W2.grad
            
            W1.grad.zero_()
            W2.grad.zero_()

# 4: Self define auto grad
def selfDefineAutoGrad():  #raw is 500
    class MyReLU( torch.autograd.Function ):
        @staticmethod                        #important
        def forward( ctx, x ):
            ctx.save_for_backward( x )
            return x.clamp( min= 0 )
        @staticmethod                        #no this can not run
        def backward( ctx, grad_output ):
            x, = ctx.saved_tensors
            grad_x = grad_output.clone()
            grad_x[x < 0] = 0
            return grad_x 
    
    device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu')

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn( N, D_in, device= device )
    y = torch.randn( N, D_out, device= device )

    W1 = torch.randn( D_in, H, device= device, requires_grad= True )
    W2 = torch.randn( H, D_out, device= device, requires_grad= True )

    learning_rate = 1e-6
    for t in range( 50 ):
        y_pred = MyReLU.apply( x.mm(W1).mm(W2) )

        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

        loss.backward()

        with torch.no_grad():
            W1 -= learning_rate * W1.grad
            W2 -= learning_rate * W2.grad
            
            W1.grad.zero_()
            W2.grad.zero_()

# 5: TensorFlow static graph. like autoGrad in pytorch

# 6: nn module
def infoNNmodule(): 
    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn( N, D_in )
    y = torch.randn( N, D_out, )

    model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out) )

    loss_fn = torch.nn.MSELoss( reduction= 'sum' )

    learning_rate = 1e-4
    for t in range( 50 ):  #raw is 500
        y_pred = model( x )

        loss = loss_fn( y_pred, y )
        print( t, loss.item() )

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

# 7: optim module
def infoOptim():
    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn( N, D_in )
    y = torch.randn( N, D_out )

    model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out) )

    loss_fn = torch.nn.MSELoss( reduction= 'sum' )

    learning_rate = 1e-4

    optimizer = torch.optim.Adam( model.parameters(), lr= learning_rate )

    for t in range( 500 ):  #raw is 500
        y_pred = model( x )

        loss = loss_fn(y_pred, y)
        print( t, loss.item() )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
      
# 8: self define nn module
def selfDefineNNmodule():
    class TwoLayerNet( torch.nn.Module ):
        def __init__( self, D_in, H, D_out ):
            super( TwoLayerNet, self ).__init__()
            self.linear1 = torch.nn.Linear( D_in, H )
            self.linear2 = torch.nn.Linear( H, D_out )

        def forward( self, x ):
            h_relu = self.linear1( x ).clamp( min= 0 )
            y_pred = self.linear2( h_relu )
            return y_pred

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn( N, D_in )
    y = torch.randn( N, D_out )

    model = TwoLayerNet( D_in, H, D_out )

    loss_fn = torch.nn.MSELoss( reduction= 'sum' )
    optimizer = torch.optim.SGD( model.parameters(), lr= 1e-4 )
    for t in range( 50 ):   #raw is 500
        y_pred = model( x )
        
        loss = loss_fn(y_pred, y)
        print( t, loss.item() )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 9: pytorch control flow
def pytorchControlFlow():
    class DynamicNet( torch.nn.Module ):
        def __init__( self, D_in, H, D_out ):
            super( DynamicNet, self).__init__()
            self.input_linear = torch.nn.Linear( D_in, H )
            self.middle_linear = torch.nn.Linear( H, H )
            self.output_linear = torch.nn.Linear( H, D_out )

        def forward( self, x ):
            h_relu = self.input_linear( x ).clamp( min= 0 )
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
    for t in range( 50 ):   #raw is 500
        y_pred = model( x )

        loss = criterion( y_pred, y )
        print( t, loss.item() )

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()      
      
if __name__ == "__main__":
    numpyInfo()       
    #tensorAcc() # not run on AGX, cuda:0 is not available
    
    #autoGrad() # not run on AGX, cuda:0 is not available
    selfDefineAutoGrad()
    
    #TensorFlow static graph
    
    infoNNmodule() 
    infoOptim()
    
    selfDefineNNmodule()
    pytorchControlFlow()