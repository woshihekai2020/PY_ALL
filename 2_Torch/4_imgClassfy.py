#特别是对于视觉，我们已经创建了一个叫做 totchvision 的包，该包含有支持加载类似
#Imagenet，CIFAR10，MNIST 等公共数据集的数据加载模块 torchvision.datasets
#和支持加载图像数据数据转换模块 torch.utils.data.DataLoader。
#这提供了极大的便利，并且避免了编写“样板代码”。
# 1: download data
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./DATA/4_data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./DATA/4_data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




# 2: show images
print("# 2: show images")
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img  = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow( np.transpose(npimg, (1, 2, 0)) )
    plt.show()

dataiter = iter( trainloader ) # get some random training images
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print( ' '.join('%5s' % classes[labels[j]] for j in range(4) ))




# 3: train a model
print("# 3: train a model")
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





# 4: define lost function and optimizer"
print("# 4: define lost function and optimizer")
import torch.optim as optim
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD( net.parameters(), lr= 0.001, momentum= 0.9 )

for epoch in  range(2):
    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net( inputs )
        loss = criterion( outputs, labels )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print( 'Finished Training\n' )




# 5: value the perference of ths model
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