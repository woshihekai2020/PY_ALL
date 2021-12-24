#download install package
from __future__ import print_function, division
from tkinter import *
import os
import torch
import pandas as pd
from skimage import io, transform  #skimage -- fullname: scikit-image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings( "ignore" )

plt.ion()
print( matplotlib.get_backend() )

landmarks_frame = pd.read_csv( '../../data/faces/face_landmarks.csv' )

n = 65
img_name = landmarks_frame.iloc[n, 0]
#landmarks = landmarks_frame.iloc[n, 1:].as_matrix() #old version
landmarks = landmarks_frame.iloc[n, 1:].values
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_landmarks( image, landmarks ):
    plt.imshow( image )
    plt.scatter( landmarks[:, 0], landmarks[:, 1], s= 10, marker= '.', c= 'r' )
    plt.pause( 0.001 )

plt.figure()
show_landmarks( io.imread( os.path.join('../../data/faces/', img_name)), landmarks )
plt.show()
#plt.pause(0) #can run. have problem

class FaceLandMarksDataset( Dataset ):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file（string）：带注释的csv文件的路径。
        root_dir（string）：包含所有图像的目录。
        transform（callable， optional）：一个样本上的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                            self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
face_dataset = FaceLandMarksDataset( csv_file= "../../data/faces/face_landmarks.csv",
                                     root_dir= "../../data/faces/" )
fig = plt.figure()
for i in range( len(face_dataset) ):
    sample = face_dataset[i]

    print( i, sample['image'].shape, sample['landmarks'].shape )

    ax = plt.subplot( 1, 4, i +1 )
    plt.tight_layout()
    ax.set_title( 'Sample #{}'.format(i) )
    ax.axis( 'off' )
    show_landmarks( **sample )

    if i == 3:
        plt.show()
        plt.pause(0)
        break










