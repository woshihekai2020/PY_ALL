#https://www.pytorch123.com/ThirdSection/TransferLearning/#1
#PyTorch之迁移学习
from __future__ import print_function, division
from operator import mod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
#实际中，没有人会从零开始（随机初始化）训练完整的卷积网络，迁移学习的两个主要场景：
#  使用预训练的网络来初始化自己的网络，而不是随机初始化。其他的训练步骤不变。
#  固定ConvNet除了最后的全连接层外的其他所有层。最后的全连接层被替换成一个新的随机初始化的层，只有新层会被训练。
#通常在一个很大的数据集上进行预训练得到卷积网络ConvNet, 然后将这个ConvNet的参数作为目标任务的初始化参数或者固定这些参数。

##################################################################################################### 1: 导入相关的包和数据
import wget #这里有11种方法，供你用Python下载文件https://zhuanlan.zhihu.com/p/587382385
rootDir = data_dir = "/0_DATA/8_data"
url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
filePath = rootDir + '/hymenoptera_data.zip'
if ( not os.path.isfile( filePath ) ):
    wget.download(url, filePath )
#python zip解压文件到指定文件夹https://blog.51cto.com/u_16175474/7867250
import zipfile
zip_path = rootDir + '/hymenoptera_data.zip'
zip_file = zipfile.ZipFile(zip_path, 'r')
extract_path = rootDir
zip_file.extractall( extract_path )
zip_file.close()

############################################################################################################## 2:加载数据
# 训练一个模型来分类蚂蚁ants和蜜蜂bees。训练120张验证75张。小数据集难以范化。迁移学习增强范化。
data_transforms ={
    'train':
        transforms.Compose([
                            transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
                            transforms.RandomHorizontalFlip(),  # 随机水平翻转
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ]),
    'val':
        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ]),
}
data_dir = rootDir + "/hymenoptera_data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
               ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################################################### 3: 可视化数据
#可视化部分训练图像,
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)
#批量制作网格训练数据
def trainOneBatchData():
    inputs, classes = next(iter(dataloaders['train']))          # 获取一批训练数据
    out = torchvision.utils.make_grid(inputs)                   # 批量制作网格
    imshow(out, title=[class_names[x] for x in classes])

############################################################################################################# 4: 训练模型
#通用函数训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:                              # 每个epoch都有一个训练和验证阶段
            if phase == 'train':
                scheduler.step()                                    # 学习速率调整类的对象
                model.train()                                       # 设置模型为训练模式
            else:
                model.eval()                                        # 设置模型为评估模式
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:               # 迭代数据.
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()                               # 参数梯度置零

                with torch.set_grad_enabled(phase == 'train'):      # 前向传播
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':                            # 后向+仅在训练阶段进行优化
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)        # 统计
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:             # 深度复制mo
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)                           # 加载最佳模型权重
    return model

################################################################################################### 5：可视化模型的预测结果
# 一个通用的展示少量预测图片的函数
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

################################################################################################# 6：使用场景1：微调ConvNet
# 加载预训练模型并重置最终完全连接的图层
def finetuningConvNet():
    model_ft = models.resnet18(pretrained=True)                                     # 加载预训练模型
    num_ftrs = model_ft.fc.in_features                                              # 获取最后一层的输入特征数
    model_ft.fc = nn.Linear(num_ftrs, 2)                                 # 替换最后一层为新的线性层
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()                                               # 损失函数
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)         # 优化器：所有参数参与优化
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    # 学习率调度器：每7个epoch学习率衰减0.1
    # 训练模型
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    # 可视化模型，评估效果
    visualize_model(model_ft)

##################################################################################### 7：使用场景2：ConvNet作为固定特征提取器
# 特征提取，加载预训练模型并冻结所有卷积层。
def extractFeatureConvNet():
    model_conv = models.resnet18(pretrained=True)                                   # 加载预训练模型
    for param in model_conv.parameters():                                           # 冻结所有卷积层
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features                                            # 获取最后一层的输入特征数
    model_conv.fc = nn.Linear(num_ftrs, 2)                               # 替换最后一层为新的线性层
    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()                                               # 损失函数
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)  # 优化器：仅最后一层参与优化
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)  # 学习率调度器：每7个epoch学习率衰减0.1
    # 训练模型
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    # 可视化模型，评估效果
    visualize_model(model_conv)

if __name__=="__main__":
    #trainOneBatchData()             #可视化部分训练图像，以便了解数据扩充。

    #finetuningConvNet()              #微调ConvNet

    extractFeatureConvNet()          #固定特征提取器：冻结除最后一层外的所有层。

    plt.ioff()
    plt.show()