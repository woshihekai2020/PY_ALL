#https://www.pytorch123.com/SecondSection/training_a_classifier/
# PyTorch 图像分类器
import os
rootDir = './DATA/4_data'
os.makedirs(rootDir, exist_ok= True)    # check dir exist or not?
#对于视觉，我们已经创建了一个叫做 totchvision 的包，该包含有支持加载类似
#Imagenet，CIFAR10，MNIST 等公共数据集的数据加载模块 torchvision.datasets 和
#支持加载图像数据数据转换模块 torch.utils.data.DataLoader。
########训练一个图像分类器:对于本教程，我们将使用CIFAR10数据集，
# 它包含十个类别：‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’。
# CIFAR-10 中的图像尺寸为33232，也就是RGB的3层颜色通道，每层通道内的尺寸为32*32。
########
import torch
import torchvision
import torchvision.transforms as transforms

# 1: 使用torchvision加载并且归一化CIFAR10的训练和测试数据集
print("1: load img data")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=rootDir, train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=rootDir, train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




# 2: 展示其中的一些训练图片
print("# 2: show images")
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):                                                                            # functions to show an image
    img  = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow( np.transpose(npimg, (1, 2, 0)) )
    plt.show()
dataiter = iter( trainloader )                                                         # get some random training images
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))                                                                # show images
print( ' '.join('%5s' % classes[labels[j]] for j in range(4) ))                                           # print labels



# 3: 定义网络
print("# 3: define a model")
import torch.nn as nn
import torch.nn.functional as F
class Net( nn. Module ):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear( 16 * 5 * 5, 120)
        self.fc2 = nn.Linear( 120, 84 )
        self.fc3 = nn.Linear( 84, 10 )
    def forward( self, x ):
        x = self.pool( F.relu(self.conv1(x)))
        x = self.pool( F.relu(self.conv2(x)))
        x = x.view( -1, 16 * 5 * 5 )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
if torch.cuda.is_available() :
    device = torch.device("cuda:0")
    print( "\n device is cuda:0 \n")
else:
    print( "\n device is cpu \n" )
    device = torch.device("cpu")
net.to(device)



# 4: 定义一个损失函数和优化器
print("# 4: define lost function and optimizer")
import torch.optim as optim
criterion = nn.CrossEntropyLoss()                                                         #类交叉熵Cross-Entropy 作损失函数。
optimizer = optim.SGD( net.parameters(), lr= 0.001, momentum= 0.9 )                                       #动量SGD做优化器。
for epoch in  range(2):                                                                                         #训练网络
    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net( inputs )                                                                                 #forward
        loss = criterion( outputs, labels )                                                                    #backward
        loss.backward()
        optimizer.step()                                                                                       #optimize

        running_loss += loss.item()
        if i % 2000 == 1999:                                                             # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print( 'Finished Training\n' )



# 5: 评估模型网络在整个数据集上的表现
print("# 5: value the perference of ths model")
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net( images )
        _, predicted = torch.max( outputs.data, 1 )
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Accuracy of the network on the 10000 test images: %d %%" %(100 * correct / total))

# 6：在每个类别上评估模型的性能
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



#特别是对于视觉，我们已经创建了一个叫做 totchvision 的包，该包含有支持加载类似
#Imagenet，CIFAR10，MNIST 等公共数据集的数据加载模块 torchvision.datasets
#和支持加载图像数据数据转换模块 torch.utils.data.DataLoader。
#这提供了极大的便利，并且避免了编写“样板代码”。