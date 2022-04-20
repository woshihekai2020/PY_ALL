#source code :https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#putting-everything-together
import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

#in folder ~/dep/1_Pytorch/py_dep/vision-0.10.0/reference/detection/
from engine import train_one_epoch, evaluate
import utils
import transforms as T
#need install pycocotools for preparing data


class PennFudanDataset( object ):
    def __init__(self, root, transforms ):
        self.root = root
        self.transforms = transforms

        self.imgs = list( sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list( sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join( self.root, "PNGImages", self.imgs[idx] )
        mask_path = os.path.join( self.root, "PedMasks", self.masks[idx] )
        img = Image.open( img_path ).convert( "RGB" )

        mask = Image.open( mask_path )
        mask = np.array( mask )

        obj_ids = np.unique( mask )
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len( obj_ids )
        boxes = []
        for i in range( num_objs ):
            pos = np.where( masks[i] )
            xmin = np.min( pos[1] )
            xmax = np.max( pos[1] )
            ymin = np.min( pos[0] )
            ymax = np.max( pos[0] )
            boxes.append( [xmin, ymin, xmax, ymax] )

        boxes = torch.as_tensor( boxes, dtype= torch.float32 )
        labels = torch.ones( (num_objs), dtype= torch.int64)
        masks = torch.as_tensor( masks, dtype= torch.uint8 )

        image_id = torch.tensor( [idx] )
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros( (num_objs, ), dtype= torch.int64 )

        target = {}
        target["boxes"]  = boxes
        target["labels"] = labels
        target["masks"]  = masks
        target["image_id"] = image_id
        target["area"]   = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, traget = self.transforms( img, target )

        return img, target

    def __len__(self):
        return len( self.imgs )


#small change
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn( pretrained= True )

    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor( in_features, num_classes )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

#big change
#backbone = torchvision.models.mobilenet_v2( pretrained= True ).features
#backbone.out_channels = 1280
#anchor_generator = AnchorGenerator( sizes= ((32, 64, 128, 256, 512), ), aspect_ratios= ((0.5, 1.0, 2.0), ))
#roi_pooler = torchvision.ops.MultiScaleRoIAlign( featmap_names= [0], output_size= 7, sampling_ratio= 2 )
#model = FasterRCNN( backbone, num_classes= 2, rpn_anchor_generator= anchor_generator, box_roi_pool= roi_pooler )

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation( num_classes ):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn( pretrained= True )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor( in_features, num_classes )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor( in_features_mask, hidden_layer, num_classes )

    return model


def get_transform( train ):
    transforms = []
    transforms.append( T.ToTensor() )
    if train:
        transforms.append( T.RandomHorizontalFlip(0.5))
    return T.Compose( transforms )

def main():
    device = torch.device( 'cuda' ) if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    dataset = PennFudanDataset( './data/PennFudanPed', get_transform(train= True) )
    dataset_test = PennFudanDataset( './data/PennFudanPed', get_transform( train= False ))

    indices = torch.randperm( len(dataset)).tolist()
    dataset = torch.utils.data.Subset( dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset( dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader( dataset, batch_size= 2, shuffle= True,
                                               num_workers= 4, collate_fn= utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader( dataset_test, batch_size= 1, shuffle= False,
                                                       num_workers= 4, collate_fn= utils.collate_fn)

    model = get_model_instance_segmentation( num_classes )
    model.to( device )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD( params, lr= 0.005, momentum= 0.9, weight_decay= 0.0005 )
    lr_scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size= 3, gamma= 0.1 )

    num_epochs = 10

    for epoch in range( num_epochs ):
        train_one_epoch( model, optimizer, data_loader, device, epoch, print_freq= 10 )
        lr_scheduler.step()
        evaluate( model, data_loader_test, device= device )
    print( "that is it")


if __name__ == "__main__":
    main()




