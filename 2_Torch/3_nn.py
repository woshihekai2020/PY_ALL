#https://www.pytorch123.com/SecondSection/neural_networks/
# PyTorch 神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F
print( "\n\n####  work_flow:\n  "
       " input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d\n",
       "       -> view -> linear -> relu -> linear -> relu -> linear \n",
       "       -> MSELoss \n",
       "       -> loss " )

########################################################################################## 1:定义一个包含训练参数的神经网络
print( "\n\n\n\n# 1:定义一个包含训练参数的神经网络" )
class Net( nn.Module ):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d( 1, 6, 5 )
        self.conv2 = nn.Conv2d( 6,16, 5 )
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear( 16 *5 * 5, 120 )
        self.fc2 = nn.Linear( 120, 84 )
        self.fc3 = nn.Linear(  84, 10 )

    def forward( self, x ):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d( F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d( F.relu(self.conv2(x)), 2)
        x = x.view( -1, self.num_flat_features(x) )
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = self.fc3(x)
        return x
    
    def num_flat_features( self, x ):
        size = x.size()[1 :]  # all dimensions except the batch dimension
        num_features =  1
        for s in size:
            num_features *= s
        return num_features
net = Net()
print( "~~this is net :\n  ", net )
print( "\n模型训练的参数通过net.parameters()调用")
params = list( net.parameters() )
print( "~~conv1's.weight=list(net.parameters())[0].size(): \n  ", params[0].size() )
print( "~~len(params): \n  " , len(params) )

##########################################################################################################2：迭代整个输入
#期望的输入维度是32x32,为使用mnist数据集，需要修改维度为32x32
print("\n\n\n\n#2:迭代整个输入:" )
print("随机生成一个32x32的输入:")
input = torch.randn( 1, 1, 32, 32)
out = net( input )
print( "~~output of net(1, 1, 32, 32) : \n  ", out )
print( "~~所有参数梯度缓存器置零，用随机的梯度来反向传播")
net.zero_grad()
#out.backward(torch.randn(1, 10)) 

################################################################################################### 3:通过神经网络处理输入
print("\n\n\n\n3:通过神经网络处理输入:" )
output = net( input )
target = torch.randn( 10 )     # a dummy target, for example
target = target.view( 1, -1 )  # make it the same shape as output

########################################################################################################### 4:计算损失值
#一个损失函数需要一对输入：模型输出和目标，然后计算一个值来评估输出距离目标有多远。
#在一些不同的损失函数在nn包中，一个简单的损失函数为nn.MSELoss，计算均方误差
print( "\n\n\n\n#4: 计算损失值") #nn module work flow
print( "定义损失函数，计算损失值。计算一个值来评估输出距离目标远。下面为均方误差MSEloss:" )
criterion = nn.MSELoss()
loss = criterion( out, target )
print( " ~~loss{ between out(net(input)) and target(random set) }: \n  ", loss )
print( "\n ~~(MSELoss)loss.grad_fn: \n  ", loss.grad_fn )
print( "\n ~~(Linear)loss.grad_fn_next_functions[0][0]: \n  ", loss.grad_fn.next_functions[0][0] )
print( "\n ~~(ReLU)loss.grad_fn.next_functions[0][0].next_functions[0][0] : \n  ",
       loss.grad_fn.next_functions[0][0].next_functions[0][0] )

############################################################################################################# 5:反向传播
# 需要清空现存的梯度，要不然将会和现存的梯度累计到一起
print( "\n\n\n\n#5:反向传播，需要清空现存的梯度，要不然将会和现存的梯度累计到一起" )
print( "zeroes the gradient buffers of all parameters: net.zero_grad()\n  " )
net.zero_grad() 

print( "con1的偏置项在反向传播前后的变化")
print( "~~before backward, net.conv1.bias.grad : " )
print( net.conv1.bias.grad ,"\n")

######################################################################################################### 6:更新网络权重
print( "\n\n\n\n#6:跟新网络权重" )
loss.backward()
print( "~~after backward, net.conv1.bias.grad :  ")
print( net.conv1.bias.grad )