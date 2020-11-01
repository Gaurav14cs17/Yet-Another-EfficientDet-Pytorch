# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

import argparse
import datetime
import os
import traceback

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import Resizer, Flip_X, Flip_Y, Normalizer, Equalize, Brightness, ComposeAlb, Constrast, collater, TobyCustom
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string

from tqdm import tqdm

compound_coef = 4
force_input_size = None  # set None to use default size
# img_path = 'test/img.png'

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False


obj_list = ['ROI']


color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=90,
                             ratios=anchor_ratios, scales=anchor_scales)
# weights_path = './weights/efficientdet-d4.pth'
# model.load_state_dict(torch.load(weights_path), strict=False)
model.backbone_net.model._conv_stem.conv = nn.Conv2d(4, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)

model.load_state_dict(torch.load('C:/Users/giang/Desktop/result/save/coco/efficientdet-d4_10_1500.pth', map_location='cpu'))
# model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

params = Params(f'projects/coco.yml')
params.num_gpus = 1
if params.num_gpus == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)

# opt.saved_path = opt.saved_path + f'/{params.project_name}/'
# opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
# os.makedirs(opt.log_path, exist_ok=True)
# os.makedirs(opt.saved_path, exist_ok=True)


val_params = {'batch_size': 1,
                'shuffle': False,
                'drop_last': True,
                'collate_fn': collater,
                'num_workers': 0}

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

root_val = 'D:/Etri_tracking_data/Etri_full/val_1024/'
side_val = 'D:/Etri_tracking_data/Etri_full/val_Sejin_1024/'
ground_truth_val = 'D:/Etri_tracking_data/Etri_full/val_1024.txt'
# root = '/home/../../data3/giangData/image_crop_1175x7680/'
# side = '/home/../../data3/giangData/image_vol1_Sejin/'
# ground_truth = '/home/../../data3/giangData/specific_train.txt'

val_set = TobyCustom(root_dir=root_val, side_dir = side_val, \
                         annot_path = ground_truth_val, \
                         transform=ComposeAlb([Resizer(input_sizes[4]),
                                               Normalizer()]))
val_generator = DataLoader(val_set, **val_params)


with torch.no_grad():
    for iter, data in enumerate(val_generator):
        # print(torch.max(imgs))
        # print(imgs.shape)
        # from matplotlib import pyplot as plt
        # image = imgs[0]
        
        # image = image[0:3,:,:]
        # mean=np.array([[[0.485, 0.456, 0.406]]])
        # mean = np.reshape(mean, mean.shape[::-1]) 
        # std = np.array([[[0.229, 0.224, 0.225]]])
        # std = np.reshape(std, std.shape[::-1])
        # image*=std
        # image+=mean
        # image*=255
        # image = image.type(torch.int32)
        # image = image.numpy()
        # image = np.einsum('abc->bca',image)
        # image = image[:,:,::-1]
        # print(torch.max(image))
        # print(torch.min(image))
        # print(image.shape)
        # print(image)
        # import cv2

        # print(image)
        # cv2.imwrite('C:/Users/giang/Desktop/some.png', image)
        imgs = data['img']
        annot = data['annot']

        if params.num_gpus == 1:
            imgs = imgs.cuda()
            annot = annot.cuda()

        _, regression, classification, anchors = model(imgs)
        # print(regression)
        print(regression.shape)
        print(classification.shape)
        print(anchors.shape)
        break
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
        print(out)
        




# with torch.no_grad():
#     for i, data in enumerate(tqdm(val_generator, position=0, leave=True)):
#         # data_batch = data.to(device)
#         print(dict)
#         data_bath = data.cuda()
                            
#         b_size = data_batch.size(0)

#         features, regression, classification, anchors = model(data_batch)

#         regressBoxes = BBoxTransform()
#         clipBoxes = ClipBoxes()

#         out = postprocess(x,
#                         anchors, regression, classification,
#                         regressBoxes, clipBoxes,
#                         threshold, iou_threshold)
#         print(out)
#         break

def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


# out = invert_affine(framed_metas, out)
# display(out, ori_imgs, imshow=False, imwrite=True)

# print('running speed test...')
# with torch.no_grad():
#     print('test1: model inferring and postprocessing')
#     print('inferring image for 10 times...')
#     t1 = time.time()
#     for _ in range(10):
#         _, regression, classification, anchors = model(x)

#         out = postprocess(x,
#                           anchors, regression, classification,
#                           regressBoxes, clipBoxes,
#                           threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)

#     t2 = time.time()
#     tact_time = (t2 - t1) / 10
#     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    # uncomment this if you want a extreme fps test
    # print('test2: model inferring only')
    # print('inferring images for batch_size 32 for 10 times...')
    # t1 = time.time()
    # x = torch.cat([x] * 32, 0)
    # for _ in range(10):
    #     _, regression, classification, anchors = model(x)
    #
    # t2 = time.time()
    # tact_time = (t2 - t1) / 10
    # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
