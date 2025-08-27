#https://www.pytorch123.com/SecondSection/optional_data_parallelism/
#RAW:https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
# PyTorch 数据并行处理
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
os.makedirs('./0_DATA/5_data', exist_ok= True)    # check dir exist or not?

############################################################################################################# 1:参数
print("1:参数") #("# 1:parameters and dataloaders")
input_size = 5
output_size = 2
batch_size = 30
data_size = 100

########################################################################################################## 2:设备选择
print("2:设备选择") #("# 2:device select")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################################################## 3:准备数据集
print("3:准备数据集") #("# 3:prepare dataset")
class RandomDataset( Dataset ):                                                                     #生成一个玩具数据。
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__( self ):
        return self.len
rand_loader = DataLoader( dataset = RandomDataset(input_size, data_size), batch_size= batch_size, shuffle= True )

########################################################################################################## 4:简单模型
print("4:简单模型") #("# 4:simple model")
class Model( nn.Module ):
    def __init__( self, input_size, output_size ):
        super( Model, self ).__init__()
        self.fc = nn.Linear( input_size, output_size )
    def forward( self, input ):
        output = self.fc( input )
        print( "\t In Model: input size", input.size(), "output size", output.size() )
        return output
model = Model( input_size, output_size )
if torch.cuda.device_count() > 1 :
    print( "Let's use " , torch.cuda.device_count(), "GPUs!" )
    model = nn.DataParallel( model )
model.to( device )

######################################################################################################### 5: 运行模型
print("5: 运行模型") #("# 5: run model")
for data in rand_loader:
    input = data.to( device )
    output = model( input )
    print( "Outside : input_size ", input.size(), "output_size", output.size() )
