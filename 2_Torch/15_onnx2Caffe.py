#https://www.pytorch123.com/FourSection/ONNX/
# 使用ONNX将模型转移至Caffe2和移动端
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init

###########################################################################################################0: 准备数据
import os
rootDir = './DATA/15_data'
os.makedirs(rootDir, exist_ok= True)    # check dir exist or not?
import wget                                     #这里有11种方法，供你用Python下载文件https://zhuanlan.zhihu.com/p/587382385
url = "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"
filePath = rootDir + '/superres_epoch100-44c6958e.pth'
if ( not os.path.isfile( filePath ) ):
    wget.download(url, filePath )

##########################################################################################################1： 引入模型
# 1.1 SuperResolution模型
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

# 1.2 训练模型
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict( model_zoo.load_url(model_url, map_location= map_location) )
torch_model.train( False )

# 1.3 导出模型
x = torch.randn( batch_size, 1, 224, 224, requires_grad= True )
torch_out = torch.onnx._export( torch_model, x, "super_resolution.onnx", export_params= True)

# 1.4 采用ONNX表示模型并在Caffe2中使用
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

#加载ONNX ModelProto对象。模型是一个标准的Python protobuf对象
model = onnx.load("super_resolution.onnx")

# 为执行模型准备caffe2后端，将ONNX模型转换为可以执行它的Caffe2 NetDef。
# 其他ONNX后端，如CNTK的后端即将推出。
prepared_backend = onnx_caffe2_backend.prepare(model)

# 在Caffe2中运行模型

# 构造从输入名称到Tensor数据的映射。
# 模型图形本身包含输入图像之后所有权重参数的输入。由于权重已经嵌入，我们只需要传递输入图像。
# 设置第一个输入。
W = {model.graph.input[0].name: x.data.numpy()}

# 运行Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# 验证数字正确性，最多3位小数
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)

print("Exported model has been executed on Caffe2 backend, and the result looks good!")

###############################################################################################2: 使用ONNX转换SRResNET


#################################################################################################3: 在移动设备上运行模型
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils
import numpy as np
import os
import subprocess
from PIL import Image
from matplotlib import pyplot
from skimage import io, transform

# 3.1 加载图像并预处理
# 加载图像
img_in = io.imread("./_static/img/cat.jpg")
# 设置图片分辨率为 224x224
img = transform.resize(img_in, [224, 224])
# 保存好设置的图片作为模型的输入
io.imsave("./_static/img/cat_224x224.jpg", img)


# 3.2 在Caffe2运行并输出
# 加载设置好的图片并更改为YCbCr的格式
img = Image.open("./_static/img/cat_224x224.jpg")
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()
# 让我们运行上面生成的移动网络，以便正确初始化caffe2工作区
workspace.RunNetOnce(init_net)
workspace.RunNetOnce(predict_net)
# Caffe2有一个很好的net_printer能够检查网络的外观
# 并确定我们的输入和输出blob名称是什么。
print(net_printer.to_string(predict_net))

# 现在，让我们传递调整大小的猫图像以供模型处理。
workspace.FeedBlob("9", np.array(img_y)[np.newaxis, np.newaxis, :, :].astype(np.float32))
# 运行predict_net以获取模型输出
workspace.RunNetOnce(predict_net)
# 现在让我们得到模型输出blob
img_out = workspace.FetchBlob("27")

img_out_y = Image.fromarray(np.uint8((img_out[0, 0]).clip(0, 255)), mode='L')
# 获取输出图像遵循PyTorch实现的后处理步骤
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
# 保存图像，我们将其与移动设备的输出图像进行比较
final_img.save("./_static/img/cat_superres.jpg")

# 3.3 在移动设备上运行模型
# 让我们先把一堆东西推到adb，指定二进制的路径
CAFFE2_MOBILE_BINARY = ('caffe2/binaries/speed_benchmark')

# 我们已经在上面的步骤中保存了`init_net`和`proto_net`，我们现在使用它们。
# 推送二进制文件和模型protos
os.system('adb push ' + CAFFE2_MOBILE_BINARY + ' /data/local/tmp/')
os.system('adb push init_net.pb /data/local/tmp')
os.system('adb push predict_net.pb /data/local/tmp')

# 让我们将输入图像blob序列化为blob proto，然后将其发送到移动设备以供执行。
with open("input.blobproto", "wb") as fid:
    fid.write(workspace.SerializeBlob("9"))

# 将输入图像blob推送到adb
os.system('adb push input.blobproto /data/local/tmp/')

# 现在我们在移动设备上运行网络，查看`speed_benchmark --help`，了解各种选项的含义
os.system(
    'adb shell /data/local/tmp/speed_benchmark '                     # binary to execute
    '--init_net=/data/local/tmp/super_resolution_mobile_init.pb '    # mobile init_net
    '--net=/data/local/tmp/super_resolution_mobile_predict.pb '      # mobile predict_net
    '--input=9 '                                                     # name of our input image blob
    '--input_file=/data/local/tmp/input.blobproto '                  # serialized input image
    '--output_folder=/data/local/tmp '                               # destination folder for saving mobile output
    '--output=27,9 '                                                 # output blobs we are interested in
    '--iter=1 '                                                      # number of net iterations to execute
    '--caffe2_log_level=0 '
)

# 从adb获取模型输出并保存到文件
os.system('adb pull /data/local/tmp/27 ./output.blobproto')


# 我们可以使用与之前相同的步骤恢复输出内容并对模型进行后处理
blob_proto = caffe2_pb2.BlobProto()
blob_proto.ParseFromString(open('./output.blobproto').read())
img_out = utils.Caffe2TensorToNumpyArray(blob_proto.tensor)
img_out_y = Image.fromarray(np.uint8((img_out[0,0]).clip(0, 255)), mode='L')
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
final_img.save("./_static/img/cat_superres_mobile.jpg")

