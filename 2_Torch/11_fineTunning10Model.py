#https://www.pytorch123.com/FourSection/FinetuningTorchVisionModel/
#微调 Torchvision 模型
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
import matplotlib.pyplot as plt
import time
import copy
import os
matplotlib.use('TkAgg')
rootDir = './DATA/11_data'
os.makedirs(rootDir, exist_ok= True)
###################################################################################################### 1: 导包并打印版本号
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

############################################################################################################## 2:下载数据
#这里有11种方法，供你用Python下载文件https://zhuanlan.zhihu.com/p/587382385
import wget
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

######################################################################################################## 3: 输入并训练模型
# 顶级数据目录。 这里我们假设目录的格式符合ImageFolder结构
data_dir = rootDir + "/hymenoptera_data"
# 从[resnet, alexnet, vgg, squeezenet, densenet, inception]中选择模型
model_name = "squeezenet"
# 数据集中类别数量
num_classes = 2
# 训练的批量大小（根据您的内存量而变化）
batch_size = 8
# 你要训练的epoch数
num_epochs = 15
# 用于特征提取的标志。 当为False时，我们微调整个模型，# 当True时我们只更新重新形成的图层参数
feature_extract = True
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print(' - ' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向,如果只在训练时则跟踪轨迹
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2

                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
        # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            besh_model_wts = copy.deepcopy(model.state_dict())
        if phase == 'val':
            val_acc_history.append(epoch_acc)
    # print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

####################################################################################################### 4: 初始化并重塑网络
# 4.1:Resnet
# 4.1:Alexnet
# 4.3:VGG
# 4.4:Squeezenet
# 4.5:Densenet
# 4.6:Inception v3
# 4.7:重塑代码
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 初始化将在此if语句中设置的这些变量。
    # 每个变量都是模型特定的。
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
 """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
 """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
 """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
 """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
 """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
 Be careful, expects (299,299) sized images and has auxiliary output
 """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 处理辅助网络
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理主要网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def ExpInitReshapeNet():
    # 在这步中初始化模型
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # 打印我们刚刚实例化的模型
    print(model_ft)

############################################################################################################# 5: 加载数据
def loadData():
    # 在这步中初始化模型
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # 数据扩充和训练规范化
    # 只需验证标准化
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # 创建训练和验证数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # 创建训练和验证数据加载器
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    # 检测我们是否有可用的GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################################################################################################ 6: 创建优化器
def createOptimizer():
    # 在这步中初始化模型
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # 将模型发送到GPU
    model_ft = model_ft.to(device)

    # 在此运行中收集要优化/更新的参数。
    # 如果我们正在进行微调，我们将更新所有参数。
    # 但如果我们正在进行特征提取方法，我们只会更新刚刚初始化的参数，即`requires_grad`的参数为True。
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # 观察所有参数都在优化
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

######################################################################################################### 7:运行训练和验证
hist = [] # for 8 use
def tainAndValidation():
    # 在这步中初始化模型
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, loadData().dataloaders_dict, criterion, createOptimizer().optimizer_ft,
                                 num_epochs=num_epochs, is_inception=(model_name == "inception"))

###################################################################################################### 8: 对比从头开始模型
#不使用迁移学习,微调与特征提取的性能在很大程度上取决于数据集.总体优于从头开始训练。
def compareModel():
    # 初始化用于此运行的模型的非预训练版本
    scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    scratch_model = scratch_model.to(device)
    scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    scratch_criterion = nn.CrossEntropyLoss()
    _, scratch_hist = train_model(scratch_model, loadData().dataloaders_dict, scratch_criterion, scratch_optimizer,
                                  num_epochs=num_epochs, is_inception=(model_name == "inception"))

    # 绘制验证精度的训练曲线与转移学习方法
    # 和从头开始训练的模型的训练epochs的数量
    ohist = []
    shist = []

    ohist = [h.cpu().numpy() for h in hist]
    shist = [h.cpu().numpy() for h in scratch_hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    plt.plot(range(1, num_epochs + 1), shist, label="Scratch")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()

if __name__=="__main__":
    ExpInitReshapeNet()
    #loadData()
    #createOptimizer()
    #tainAndValidation()
    #compareModel()
