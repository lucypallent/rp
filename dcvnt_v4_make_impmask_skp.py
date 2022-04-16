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
from skimage import measure
import cv2

random.seed(0); torch.manual_seed(0); np.random.seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

readvdnames = lambda x: open(x).read().rstrip().split('\n')
pe_list = readvdnames(f"dataset3/NCOV-BF/ImageSets-old/lung_test.txt")[::-1]

# where saving
des_home = 'dataset4/NCOV-BF/NpyData-size224x336-impdlmask-skip'

# where loading from
src_home = 'dataset4/NCOV-BF/NpyData-size224x336'

def imp_skip_masks(x):        # dtype is "PE"/"NORMAL"
    print (x)
    if os.path.isfile(os.path.join(des_home, x+".npy")) is True:
        return
    ### raw_imgs = np.uint8(np.load(os.path.join(src_home, x+".npy")))
    raw_imgs = np.load(os.path.join(src_home, x+".npy"))
    raw_masks = np.load(os.path.join(src_home, x+"-dlmask.npy"))

    # code for improved images
    length = len(raw_masks)

    imp_masks = np.zeros((length, 224, 336))
    kernel = np.ones((5,5), np.uint8) # dilating by 5 not 10 as img is smaller than 512 x 512

    for i in range(length):
        # select the two largest shapes as the lungs
        labels_mask = measure.label(raw_masks[i])
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions[2:]:
                labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
        labels_mask[labels_mask!=0] = 1
        labels_slice = labels_mask.astype('uint8')

        # dilate the lungs by 5 pixels
        dilated_slice = cv2.dilate(labels_slice, kernel, iterations=1)

        # save new mask
        imp_masks[i] = dilated_slice

    skipped_imgs = raw_imgs[::2]
    skipped_imp_masks = imp_masks[::2]

    np.save(os.path.join(des_home, x+".npy"), skipped_imgs)
    np.save(os.path.join(des_home, x+"-dlmask.npy"), skipped_imp_masks)

from concurrent import futures

num_threads=10

with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(imp_skip_masks, x, ) for x in pe_list[::-1]]
    for i, f in enumerate(futures.as_completed(fs)):
        print ("{}/{} done...".format(i, len(fs)))
############################ end of preprocessing .npys (creating d4)
