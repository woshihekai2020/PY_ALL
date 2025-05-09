#https://www.pytorch123.com/ThirdSection/LearningPyTorch/
#PyTorch之小试牛刀
import numpy as np
import torch
import random

# 本节使用全链接的ReLU网络作为运行示例.使用单一隐藏层,进行梯度下降训练.通过最小化网络输出和结果的欧几里德距离,来拟合随机生成的数据.
#################################################################################################### 1: 核心两个主要特征
#一个n维张量，类似于numpy，但可以在GPU上运行
#搭建和训练神经网络时的自动微分/求导机制

############################################################################################################### 2:张量
# 2.1:热身:Numpy
def numpyInfo():
    #科学计算通用框架,自行构建梯度,计算图,深度学习.手动前向和反向传播,拟合随机数据.
    print(" 2.1: Numpy\n")
    N, D_in, H, D_out = 64, 1000, 100, 10           # N是批量大小; D_in是输入维度; H是隐藏层维度; D_out是输出维度
    x  = np.random.randn( N,  D_in )                  # 创建随机输入和输出数据
    y  = np.random.randn( N, D_out )
    W1 = np.random.randn( D_in,  H )                 # 随机初始化权重
    W2 = np.random.randn( H, D_out )

    learing_rate = 1e-6
    for t in range( 500 ):
        h = x.dot( W1 )                             # 前向传递：计算预测值y
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot( W2 )

        loss = np.square(y_pred - y).sum()          # 评估损失:计算和打印损失loss
        print(t, float(loss) )

        grad_y_pred = 2.0 * (y_pred - y)            # 反向传播:计算w1和w2对loss的梯度
        grad_w2 = h_relu.T.dot( grad_y_pred )
        grad_h_relu = grad_y_pred.dot( W2.T )
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot( grad_h )

        W1 -= learing_rate * grad_w1                # 更新权重:根据学习率
        W2 -= learing_rate * grad_w2
   
# 2.2: Pytorch:张量
def tensorAcc():
    # pytorch概念，类似于numpy，但可以在GPU上运行，获取50倍以上加速。
    print(" 2.2: tensorACC\n")
    dtype  = torch.float
    device = torch.device( "cuda:0")

    N, D_in, H, D_out = 64, 1000, 100, 10   # N是批量大小; D_in是输入维度; H是隐藏层维度; D_out是输出维度
    x  = torch.randn( N,  D_in, device= device, dtype= dtype )    #创建随机输入和输出数据
    y  = torch.randn( N, D_out, device= device, dtype= dtype )
    W1 = torch.randn( D_in,  H, device= device, dtype= dtype )   # 随机初始化权重
    W2 = torch.randn( H, D_out, device= device, dtype= dtype )

    learning_rate = 1e-6
    for t in range( 500 ):
        h = x.mm( W1 )                                          # 前向传递：计算预测y
        h_relu = h.clamp( min= 0 )
        y_pred = h_relu.mm( W2 )

        loss = (y_pred - y).pow(2).sum().item()                 # 评估损失:计算和打印损失
        #print(t, dtype(loss) )
        print(t, loss)

        grad_y_pred = 2.0 * (y_pred - y)                        # 反向传播:方向传播计算w1和w2相对于损耗的梯度
        grad_w2 = h_relu.t().mm( grad_y_pred )
        grad_h_relu = grad_y_pred.mm( W2.t() )
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm( grad_h )

        W1 -= learning_rate * grad_w1                           # 更新权重:使用梯度下降更新权重
        W2 -= learning_rate * grad_w2

########################################################################################################### 3：自动求导
# 3.1:Pytorch:张量和自动求导,# not run on AGX
def autoGrad():
    #调用库函数,反向传播,自动计算梯度.
    print(" 3.1: autoGrad\n")
    dtype = torch.float
    device = torch.device( "cuda:0" )

    N, D_in, H, D_out = 64, 1000, 100, 10   # N是批量大小; D_in是输入维度; H是隐藏层维度; D_out是输出维度
    x = torch.randn( N, D_in, device= device, dtype= dtype, requires_grad= True )   #创建随机输入和输出数据
    y = torch.randn( N, D_out, device= device, dtype= dtype, requires_grad= True )  #True:需要计算梯度
    W1 = torch.randn( D_in, H, device= device, dtype= dtype, requires_grad= True )
    W2 = torch.randn( H, D_out, device= device, dtype= dtype, requires_grad= True )

    learning_rate = 1e-6
    for t in range( 500 ):
        y_pred = x.mm( W1 ).clamp( min= 0 ).mm( W2 )                                #前向传递：计算预测y

        loss = (y_pred - y).pow( 2 ).sum()                                          #评估损失:计算和打印损失张量
        print( t,  loss.item() )

        # 使用autograd计算反向传播。这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
        loss.backward()                                                             #反向传播:调用库函数实现

        # 使用梯度下降更新权重。对于这一步，我们只想对w1和w2的值进行原地改变；不想为更新阶段构建计算图，
        # 所以我们使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
        with torch.no_grad():                                                       # 更新权重:使用梯度下降更新权重
            W1 -= learning_rate * W1.grad
            W2 -= learning_rate * W2.grad
            
            W1.grad.zero_()                                                         # 反向传播后手动将梯度设置为零
            W2.grad.zero_()

# 3.2: PyTorch：定义新的自动求导函数
def selfDefineAutoGrad():
    print(" 3.2: selfDefineAutoGrade\n")
    # 建立torch.autograd的子类来实现我们自定义的autograd函数，完成张量的正向和反向传播。
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

    N, D_in, H, D_out = 64, 1000, 100, 10   # N是批量大小; D_in是输入维度; H是隐藏层维度; D_out是输出维度
    x = torch.randn( N, D_in, device= device )                          # 产生输入和输出的随机张量
    y = torch.randn( N, D_out, device= device )
    W1 = torch.randn( D_in, H, device= device, requires_grad= True )    # 产生随机权重的张量
    W2 = torch.randn( H, D_out, device= device, requires_grad= True )

    learning_rate = 1e-6
    for t in range( 500 ):
        y_pred = MyReLU.apply( x.mm(W1).mm(W2) )                        # 正向传播：调用自定义MyReLU.apply 函数计算输出值y

        loss = (y_pred - y).pow(2).sum()                                # 评估损失:计算和打印损失
        print(t, loss.item())

        loss.backward()                                                 # 反向传播:调用库函数实现

        with torch.no_grad():                                           # 更新权重:使用梯度下降更新权重
            W1 -= learning_rate * W1.grad
            W2 -= learning_rate * W2.grad
            
            W1.grad.zero_()                                             # 在反向传播之后手动清零梯度
            W2.grad.zero_()

# 3.3: 静态图与动态图：TensorFlow static graph. like autoGrad in pytorch
# pytorch与tensorflow的区别在于，pytorch是动态图，tensorflow是静态图。本例使用tensorflow，无法运行。

############################################################################################################# 4:nn模块
# 4.1 PyTorch：nn
def infoNNmodule():
    # autograde过于底层, nn包中定义一组大致等价于层的模块。输入tensor输出tensor。也定义了一组损失函数，用以训练神经网络。
    print(" 4.1: use NN module in pytorch\n")
    N, D_in, H, D_out = 64, 1000, 100, 10                      # N是批量大小; D_in是输入维度; H是隐藏层维度; D_out是输出维度
    x = torch.randn( N, D_in )                                 # 输入
    y = torch.randn( N, D_out )                                # 输出

    # 使用nn包将我们的模型定义为一系列的层。按顺序应用这些模块来产生其输出。
    model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out) )

    # 设置reduction='sum'，表示我们计算的是平方误差的“和”，而不是平均值。实际使用均方误差（elementwise_mean)
    loss_fn = torch.nn.MSELoss( reduction= 'sum' )

    learning_rate = 1e-4
    for t in range( 50 ):  #raw is 500
        y_pred = model( x )                                     # 前向传播：通过向模型传入x计算预测的y。

        loss = loss_fn( y_pred, y )                             # 评估损失:计算和打印损失
        print( t, loss.item() )                                 # 传递包含y的预测值和真实值的张量，损失函数返回包含损失的张量

        model.zero_grad()                                       # 反向传播:前清零梯度

        loss.backward()                                         # 反向传播：调库函数计算模型的损失对所有可学习参数的导数（梯度）

        with torch.no_grad():                                   # 更新权重。每个参数都是张量。
            for param in model.parameters():
                param -= learning_rate * param.grad

# 4.2: Pytorch：optim
def infoOptim():
    # 对于随机梯度下降(SGD）可手动改变参数张量来更新模型权重。实践中使用复杂优化器训练。
    print(" 4.2: use optim in pytorch\n")
    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn( N,  D_in )
    y = torch.randn( N, D_out )

    model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out) )

    loss_fn = torch.nn.MSELoss( reduction= 'sum' )

    learning_rate = 1e-4

    # 使用optim包定义优化器，使用Adam优化，第一个参数是模型的更新参数
    optimizer = torch.optim.Adam( model.parameters(), lr= learning_rate )

    for t in range( 500 ):  #raw is 500
        y_pred = model( x )                         # 前向传播：通过像模型输入x计算预测的y

        loss = loss_fn(y_pred, y)                   # 评估损失:计算和打印损失张量

        optimizer.zero_grad()                       #         在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零

        loss.backward()                             # 反向传播：根据模型的参数计算loss的梯度

        optimizer.step()                            # 更新权重:调用Optimizer的step函数使它所有参数更新
      

# 4.3: PyTorch：自定义nn模块
def selfDefineNNmodule():
    #有时候需要指定比现有模块序列更复杂的模型；可继承nn模块重写forward函数.输入输出tensor
    print(" 4.3: use self Define NN module\n")
    class TwoLayerNet( torch.nn.Module ):
        def __init__( self, D_in, H, D_out ):
            super( TwoLayerNet, self ).__init__()
            self.linear1 = torch.nn.Linear( D_in,  H )
            self.linear2 = torch.nn.Linear( H, D_out )

        def forward( self, x ):
            h_relu = self.linear1( x ).clamp( min= 0 )
            y_pred = self.linear2( h_relu )
            return y_pred

    N, D_in, H, D_out = 64, 1000, 100, 10   # N是批量大小; D_in是输入维度; H是隐藏层维度; D_out是输出维度

    x = torch.randn( N, D_in )
    y = torch.randn( N, D_out )

    model = TwoLayerNet( D_in, H, D_out )
    loss_fn = torch.nn.MSELoss( reduction= 'sum' )
    optimizer = torch.optim.SGD( model.parameters(), lr= 1e-4 )
    for t in range( 50 ):   #raw is 500
        y_pred = model( x )                                         # 前向传播：通过向模型传递x计算预测值y
        
        loss = loss_fn(y_pred, y)                                   # 评估损失:计算并输出loss
        print( t, loss.item() )

        optimizer.zero_grad()                                       #        梯度清零
        loss.backward()                                             # 反向传播
        optimizer.step()                                            # 更新权重

# 4.4: PyTorch：控制流和权重共享
def pytorchControlFlow():
    # 通过在定义转发时多次重用同一个模块来实现最内层之间的权重共享。pytorch支持控制流和权重共享。可以在模型的前向传播中使用python控制流。
    print(" 4.4: use pytorch to control flow by yourself\n")
    class DynamicNet( torch.nn.Module ):
        # 三个nn.Linear实例，它们将在前向传播时被使用。
        def __init__( self, D_in, H, D_out ):
            super( DynamicNet, self).__init__()
            self.input_linear = torch.nn.Linear( D_in, H )
            self.middle_linear = torch.nn.Linear( H, H )
            self.output_linear = torch.nn.Linear( H, D_out )
        # 对于模型的前向传播，我们随机选择0、1、2、3，并重用了多次计算隐藏层的middle_linear模块。
        def forward( self, x ):
            h_relu = self.input_linear( x ).clamp( min= 0 )
            for _ in range( random.randint(0, 3) ):
                h_relu = self.middle_linear( h_relu ).clamp( min= 0 )
            y_pred = self.output_linear( h_relu )
            return y_pred

    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn( N, D_in )
    y = torch.randn( N, D_out )

    model = DynamicNet( D_in, H, D_out )                                        # 实例化上面定义的类来构造我们的模型
    criterion = torch.nn.MSELoss( reduction= 'sum' )                            # 构造我们的损失函数
    optimizer = torch.optim.SGD( model.parameters(), lr= 1e-4, momentum= 0.9 )  # 构造优化器
    for t in range( 50 ):   #raw is 500
        y_pred = model( x )

        loss = criterion( y_pred, y )
        print( t, loss.item() )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      
      
if __name__ == "__main__":
    numpyInfo()                # 2.1: numpy
    tensorAcc()                # 2.2: tensor
    
    autoGrad()                  # 3.1: 张量与自动求导
    selfDefineAutoGrad()       # 3.2: 定义新的自动求导函数
    
    #TensorFlow static graph    # ts使用静态图,pytorch使用动态图,区别是什么?
    
    infoNNmodule()             # 4.1: pytorch的nn模块
    infoOptim()                # 4.2: pytorch的optim模块
    
    selfDefineNNmodule()       # 4.3: 自定义nn模块
    pytorchControlFlow()       # 4.4: 控制流与权重共享