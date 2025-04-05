#https://www.pytorch123.com/FourSection/ONNX/
# 使用ONNX将模型转移至Caffe2和移动端
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

#1: 准备数据
import os
rootDir = './DATA/15_data'
os.makedirs(rootDir, exist_ok= True)    # check dir exist or not?
import wget                                      #这里有11种方法，供你用Python下载文件https://zhuanlan.zhihu.com/p/587382385
url = "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"
filePath = rootDir + '/superres_epoch100-44c6958e.pth'
if ( not os.path.isfile( filePath ) ):
    wget.download(url, filePath )


#2： 引入模型
class SuperResolutionNet( nn.Module ):
    def __init__(self, upscale_factor, inplace= False ):
        super(SuperResolutionNet, self).__init__()
        self.relu = nn.ReLU( inplace= inplace )
        self.conv1 = nn.Conv2d( 1, 64, (5, 5), (1, 1), (2, 2) )
        self.conv2 = nn.Conv2d( 64, 64, (3, 3), (1, 1), (1, 1) )
        self.conv3 = nn.Conv2d( 64, 32, (3, 3), (1, 1), (1, 1) )
        self.conv4 = nn.Conv2d( 32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1) )
        self.pixel_shuffle = nn.PixelShuffle( upscale_factor )
        self._initialize_weights()
    def forward( self, x ):
        x = self.relu( self.conv1(x) )
        x = self.relu( self.conv2(x) )
        x = self.relu( self.conv3(x) )
        x = self.pixel_shuffle( self.conv4(x) )
        return x
    def _initialize_weights( self ):
        init.orthogonal_( self.conv1.weight, init.calculate_gain('relu') )
        init.orthogonal_( self.conv2.weight, init.calculate_gain('relu') )
        init.orthogonal_( self.conv3.weight, init.calculate_gain('relu') )
        init.orthogonal_( self.conv4.weight )
torch_model = SuperResolutionNet( upscale_factor= 3 )


#3:  训练模型
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict( model_zoo.load_url(model_url, map_location= map_location) )
torch_model.train( False )


#4: 导出模型
x = torch.randn( batch_size, 1, 224, 224, requires_grad= True )
torch_out = torch.onnx._export( torch_model, x, "super_resolution.onnx", export_params= True)


#5: 采用ONNX表示模型并在Caffe2中使用
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

model = onnx.load("super_resolution.onnx")                      #加载ONNX ModelProto对象。模型是一个标准的Python protobuf对象

# 其他ONNX后端，如CNTK的后端即将推出。
prepared_backend = onnx_caffe2_backend.prepare(model)     # 为执行模型准备caffe2后端，将ONNX模型转换为可以执行它的Caffe2 NetDef。
# 在Caffe2中运行模型
# 构造从输入名称到Tensor数据的映射。
# 模型图形本身包含输入图像之后所有权重参数的输入。由于权重已经嵌入，我们只需要传递输入图像。
# 设置第一个输入。
W = {model.graph.input[0].name: x.data.numpy()}

c2_out = prepared_backend.run(W)[0]                                                                     # 运行Caffe2 net:

np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)                # 验证数字正确性，最多3位小数
print("Exported model has been executed on Caffe2 backend, and the result looks good!")
