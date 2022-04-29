#神经网络可以通过 torch.nn 包来构建。
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net( nn.Module ):
    def __init__(self):
        super(Net, self).__init__()
        #kernel
        # input image channel, 6:output channels, 5x5: square convolution
        self.conv1 = nn.Conv2d( 1, 6, 5 )
        self.conv2 = nn.Conv2d( 6,16, 5 )
        #an affine operation: y = Wx + b
        self.fc1 = nn.Linear( 16 * 5 * 5, 120 )
        self.fc2 = nn.Linear( 120, 84 )
        self.fc3 = nn.Linear( 84, 10 )

    def forward(self, x):
        #max pooling over a (2, 2) window
        x = F.max_pool2d( F.relu(self.conv1(x)), (2, 2))
        #if the size is a square you can only specify a single number
        x = F.max_pool2d( F.relu(self.conv2(x)), 2 )
        x = x.view( -1, self.num_flat_features(x) )
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1 :] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print( net )
#一个模型可训练的参数可以通过调用 net.parameters() 返回
params = list( net.parameters() )
print( len(params) )
print( params[0].size() )  # conv1's .weight

#随机生成一个 32x32 的输入
input = torch.randn( 1, 1, 32, 32 )
out = net( input )
print( out )

#把所有参数梯度缓存器置零，用随机的梯度来反向传播
net.zero_grad()
out.backward( torch.randn(1, 10) )



#在此，我们完成了：1.定义一个神经网络.2.处理输入以及调用反向传播
output = net( input )

#还剩下：1.计算损失值 2.更新网络中的权重
target = torch.randn( 10 )         # a dummy target, for example
target = target.view( 1, -1 )      # make it the same shape as output
#一个损失函数需要一对输入：模型输出和目标，然后计算一个值来评估输出距离目标有多远。
#有一些不同的损失函数在 nn 包中。一个简单的损失函数就是 nn.MSELoss ，均方误差。
criterion = nn.MSELoss()
loss = criterion( output, target )
print( loss )
print( "\n")
print( loss.grad_fn )    # MSELoss
print( loss.grad_fn.next_functions[0][0] )    # Linear
print( loss.grad_fn.next_functions[0][0].next_functions[0][0] ) # ReLU

# zeroes the gradient buffers of all parameters
net.zero_grad()
print( '\n conv1.bias.grad before backward' )
print( net.conv1.bias.grad )

loss.backward()
print( "conv1.bias.grad after backward" )
print( net.conv1.bias.grad )

#更新神经网络参数：最简单的更新规则就是随机梯度下降。
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD( net.parameters(), lr= 0.01 )
# in your training loop:
optimizer.zero_grad()  # zero the gradient buffers
output = net( input )
loss = criterion( output, target )
loss.backward()
optimizer.step()  # Does the update












