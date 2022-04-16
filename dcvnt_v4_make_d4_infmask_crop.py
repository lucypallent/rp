############################ start of import
import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import cv2
from PIL import Image
import random
import numpy as np
import torch

import torchvision.transforms.functional as TF

from torch.utils import data
from PIL import Image
import os
import torchvision.transforms.functional as TF
import numpy as np
import torch
import random
from scipy.ndimage import zoom
import neptune.new as neptune


random.seed(0); torch.manual_seed(0); np.random.seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

############################ start of functions for preprocessing .npys create 224x336
### = just removed
import os
import numpy as np

from scipy.ndimage import zoom

readvdnames = lambda x: open(x).read().rstrip().split('\n')

src_home = 'dataset4/NCOV-BF/NpyData-infmask'
des_home = 'dataset4/NCOV-BF/NpyData-size224x336-infmask1010-crop2'

os.makedirs(des_home, exist_ok=True)

#for d in dirs:
#    os.makedirs(os.path.join(des_home, d), exist_ok=True)

pe_list = readvdnames(f"dataset3/NCOV-BF/ImageSets-old/lung_test.txt")[::-1]

new_size = (224, 336)   # 224x336       # average isotropic shape: 193x281

new_height, new_width = new_size

clip_range = (0.15, 1)

slice_resolution = 1
from zqlib import imgs2vid
import cv2

def resize_cta_images(x):        # dtype is "PE"/"NORMAL"
    print (x)
    if os.path.isfile(os.path.join(des_home, x+".npy")) is True:
        return
    ### raw_imgs = np.uint8(np.load(os.path.join(src_home, x+".npy")))
    raw_imgs = np.load(os.path.join(src_home, x+"-masked.npy"))
    orig_imgs = np.load(os.path.join(src_home, x+"-dlmask.npy"))
    raw_masks = np.load(os.path.join(src_home, x+"-infmask.npy"))
    length = len(raw_imgs)

#     clip_imgs = raw_imgs[int(length*clip_range[0]):int(length*clip_range[1])]
#     ##
#     clip_masks = raw_masks[int(length*clip_range[0]):int(length*clip_range[1])]
#
#     raw_imgs = clip_imgs
#     raw_masks = clip_masks
#     ##
# ##c
    crop_boxes = np.zerox(shape=(length, 2, 2))
    for i, ri in enumerate(orig_imgs):
        yy, xx = np.where(ri)
    # zz, yy, xx = np.where(raw_imgs)
        crop_boxes[i] = np.array([[np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])

    crop_box = np.array([[np.min(crop_boxes[:, 0, 0]), np.max(crop_boxes[:, 0, 1])],
                         [np.min(crop_boxes[:, 1, 0]), np.max(crop_boxes[:, 1, 1])]])

    crop_imgs = raw_imgs[:, cropbox[0, 0]:cropbox[0, 1], cropbox[1, 0]:cropbox[1, 1]]

    crop_imgs = raw_masks[:, cropbox[0, 0]:cropbox[0, 1], cropbox[1, 0]:cropbox[1, 1]]

    raw_imgs = crop_imgs
    raw_masks = crop_masks
####
    height, width = raw_imgs.shape[1:3]
    zoomed_imgs = zoom(raw_imgs, (slice_resolution, new_height/height, new_width/width))
    np.save(os.path.join(des_home, x+".npy"), zoomed_imgs)
    zoomed_masks = zoom(raw_masks, (slice_resolution, new_height/height, new_width/width))
    np.save(os.path.join(des_home, x+"-dlmask.npy"), zoomed_masks)

    #imgs2vid(immasks, "debug/{}.avi".format(x))
############################ end of functions for preprocessing .npys (creating d4)

############################ start of preprocessing .npys (creating d4)
from concurrent import futures

num_threads=10

with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(resize_cta_images, x, ) for x in pe_list[::-1]]
    for i, f in enumerate(futures.as_completed(fs)):
        print ("{}/{} done...".format(i, len(fs)))
############################ end of preprocessing .npys (creating d4)
