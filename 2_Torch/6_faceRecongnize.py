#https://www.pytorch123.com/ThirdSection/DataLoding/
#PyTorch之数据加载和处理
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")
plt.ion()

#1: 准备数据
import os
rootDir = './DATA/6_data'
os.makedirs(rootDir, exist_ok= True)    # check dir exist or not?
import wget #这里有11种方法，供你用Python下载文件https://zhuanlan.zhihu.com/p/587382385
url = "https://download.pytorch.org/tutorial/faces.zip"
filePath = rootDir + "/faces.zip"
if not os.path.isfile(filePath):
    wget.download(url, filePath)
import zipfile  #python zip解压文件到指定文件夹https://blog.51cto.com/u_16175474/7867250
zip_path = filePath
zip_file = zipfile.ZipFile(zip_path, 'r')
extract_path = rootDir
zip_file.extractall( extract_path )
zip_file.close()

# 2: 展示数据集
def show_landmarks( image, landmarks ):                                                #展示一张图片和它对应的标注点作为例子。
    plt.imshow( image )
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(3)
# 3：读取数据集
def readDataSet():                                                  #将csv中的标注点数据读入（N，2）数组中，其中N是特征点的数量。
    landmarks_frame = pd.read_csv( rootDir + "/faces/face_landmarks.csv")
    n = 65
    img_name = landmarks_frame.iloc[n, 0]
   #landmarks = landmarks_frame.iloc[n, 1:].as_matrix() #old version
    landmarks = landmarks_frame.iloc[n, 1:].values
    landmarks = landmarks.astype('float').reshape(-1, 2)
    print( 'Image name: {}'.format(img_name) )
    print( 'Landmarks shape: {}'.format( img_name ) )
    print( 'First 4 Landmarks {}'.format(landmarks[: 4]) )
    plt.figure()
    show_landmarks( io.imread(os.path.join(rootDir+"/faces/", img_name)),landmarks)
    plt.show()


# 2: 为面部数据集创建一个数据集类。landmark face
class FaceLandmarksDataset( Dataset ):                                                              #"""面部标记数据集."""
    def __init__( self, csv_file, root_dir, transform= None ):
        self.landmarks_frame = pd.read_csv( csv_file )
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len( self.landmarks_frame )
    def __getitem__(self, idx):
        img_name = os.path.join( self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread( img_name )
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array( [landmarks] )
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform( sample )

        return sample


# 3: 数据可视化
def visiualizeData():                                                                           #：实例化这个类并遍历数据样本。
    face_dataset = FaceLandmarksDataset( csv_file= rootDir + "/faces/face_landmarks.csv",
                                         root_dir= rootDir + '/faces/')
    fig = plt.figure()
    for i in range( len(face_dataset)):
        sample = face_dataset[i]
        print(i, sample['image'].shape, sample['landmarks'].shape )

        ax = plt.subplot( 1, 4, i + 1 )
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks( **sample)

        if i == 3 :
            plt.show()
            break
        
        
# 4: 数据变换:                                                       绝大多数神经网络都假定图片的尺寸相同。因此我们需要做一些预处理。
class Rescale( object ):                                                                                        #缩放图片
    def __init__(self, output_size):
        assert isinstance( output_size, (int, tuple) )
        self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        h, w = image.shape[: 2]
        if isinstance( self.output_size, int ):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
               
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int( new_h ), int( new_w )

        img = transform.resize( image, (new_h, new_w))

        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop( object ):                                                                             #对图片进行随机裁剪。
    def __init__(self, output_size):
        assert isinstance( output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):                                                                    #把numpy格式图片转为torch格式图片。
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}

def transformGroup():                                         #把图像的短边调整为256，然后随机裁剪(randomcrop)为224大小的正方形。
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),RandomCrop(224)])

    fig = plt.figure()
    face_dataset = FaceLandmarksDataset( csv_file= rootDir+'/faces/face_landmarks.csv',root_dir= rootDir+'/faces/')
    sample = face_dataset[65]
    for i, tsfrm in enumerate( [scale, crop, composed]):                                        # 在样本上应用上述的每个变换。
        transformed_sample = tsfrm( sample )

        ax = plt.subplot( 1, 3, i + 1 )
        plt.tight_layout()
        ax.set_title( type(tsfrm).__name__ )
        show_landmarks( **transformed_sample )
    plt.show( )        
   
   
# 5:迭代数据集
def iterShowDataset():                                                             #把这些整合起来以创建一个带组合转换的数据集。
    transformed_dataset = FaceLandmarksDataset( 
                            csv_file = rootDir+'/faces/face_landmarks.csv',
                            root_dir = rootDir+'/faces/',
                            transform= transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    for i in range( len(transformed_dataset) ):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['landmarks'].size())
        #print( i, sample['image'].size() )
        if i == 3:
            break
 
 
# 6:批量迭代数据集                                        简单使用for循环牺牲了许多，使用多线程multiprocessingworker 并行加载数据。
def batchIterShowDataset():
    transformed_dataset = FaceLandmarksDataset( 
                            csv_file = rootDir+'/faces/face_landmarks.csv',
                            root_dir = rootDir+'/faces/',
                            transform= transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    dataLoader = DataLoader( transformed_dataset, batch_size= 4,shuffle= True, num_workers= 4 )
    def show_landmarks_batch( sample_batched ):                                                        # 辅助功能：显示批次。
        images_batch, landmarks_batch= sample_batched['image'], sample_batched['landmarks']
        batch_size = len( images_batch )
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid( images_batch )
        plt.imshow( grid.numpy().transpose((1, 2, 0)))

        for i in range( batch_size ):
            plt.scatter( landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                landmarks_batch[i, :, 1].numpy() + grid_border_size, s= 10, marker= '.', c= 'r' )
            plt.title( 'Batch from dataloader' )

    for i_batch, sample_batched in enumerate( dataLoader ):
        print( i_batch, sample_batched['image'].size(), sample_batched['image'].size() )
        if i_batch == 3:                                                                                #观察第4批次并停止。
            plt.figure()
            show_landmarks_batch( sample_batched )        
            plt.axis('off')
            plt.show()
            plt.pause( 3 )
            break
  
  
if __name__ == "__main__":
    readDataSet()
    visiualizeData()
    transformGroup()
    iterShowDataset()
    batchIterShowDataset()      