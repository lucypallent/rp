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

###### start of making infnet form -2.npy in unet-results

# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.
First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

### = just removed
import os
import numpy as np

from scipy.ndimage import zoom

readvdnames = lambda x: open(x).read().rstrip().split('\n')

src_home = 'dataset4/NCOV-BF/NpyData-size224x336-infmask1010-test'
des_home = 'dataset4/NCOV-BF/NpyData-size224x336-infmask1010-test2'

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
    ### raw_imgs = np.uint8(np.load(os.path.join(src_home, x+".npy")))
    raw_imgs = np.load(os.path.join(src_home, x+".npy"))
    raw_masks = np.load(os.path.join(src_home, x+"-dlmask.npy"))
    length = len(raw_imgs)

    zoomed_imgs = ((raw_imgs - np.min(raw_imgs)) / (np.max(raw_imgs) - np.min(raw_imgs))).astype(np.float32)

    raw_masks[raw_masks > 0.01] = 1.0
    raw_masks[raw_masks <= 0.01] = 0.0
    zoomed_masks = raw_masks.astype(np.float32)

    np.save(os.path.join(des_home, x+".npy"), zoomed_imgs)
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
