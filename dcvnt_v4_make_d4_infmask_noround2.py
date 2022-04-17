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

# create the infnet limited to being the shape of the lungs
from zqlib import imgs2vid
import cv2
import os
import numpy as np
import skimage
from skimage import measure

readvdnames = lambda x: open(x).read().rstrip().split('\n')
pe_list = readvdnames(f"d6/image_sets/all_patients.txt")[::-1]

des_home = 'dataset4/NCOV-BF/NpyData-infmask-test-1pc' # '/content/test'
src_home = 'unet-results' # '/content' # where lung masks are saved 'unet-results'

def create_masked_lungs(x):
    print(x)
    raw_imgs = np.load(os.path.join(des_home, x+'-infmask-orig-noround.npy')) # -2 is the img which appears like normal orig is blck sqr
    raw_masks = np.load(os.path.join(src_home, x+'-dlmask.npy'))

    length = len(raw_imgs)

    raw_infmasked10 = np.zeros((length, 512, 512))
    raw_masked10 = np.zeros((length, 512, 512))

    kernel = np.ones((10,10), np.uint8)

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

        # dilate the lungs by 10 pixels
        dilated_slice = cv2.dilate(labels_slice, kernel, iterations=1)

        # dilate the covid lung infection selections by 10 pixels
        dilated_img = cv2.dilate(raw_imgs[i], kernel, iterations=1)

        # mask the images
        raw_infmasked10[i] = cv2.bitwise_and(dilated_img, dilated_img, mask=dilated_slice)
        raw_masked10[i] = dilated_slice

    np.save(os.path.join(des_home, x+"-infmask-noround.npy"), raw_infmasked10)
    np.save(os.path.join(des_home, x+"-dlmask.npy"), raw_masked10)
    # np.save(os.path.join(des_home, x+".npy"), raw_imgs)

############################ start of preprocessing .npys (creating d4)
from concurrent import futures

num_threads=10

with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(create_masked_lungs, x, ) for x in pe_list[::-1]]
    for i, f in enumerate(futures.as_completed(fs)):
        print ("{}/{} done...".format(i, len(fs)))
############################ end of preprocessing .npys (creating d4)

############################ start of functions for preprocessing .npys create 224x336
### = just removed
import os
import numpy as np

from scipy.ndimage import zoom

readvdnames = lambda x: open(x).read().rstrip().split('\n')

src_home = 'dataset4/NCOV-BF/NpyData-infmask-test-1pc'
des_home = 'dataset4/NCOV-BF/NpyData-size224x336-infmask1010-test-noround'

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
    raw_imgs = np.load(os.path.join(src_home, x+"-masked.npy"))
    raw_masks = np.load(os.path.join(src_home, x+"-infmask-noround.npy"))
    length = len(raw_imgs)

    height, width = raw_imgs.shape[1:3]
    zoomed_imgs = zoom(raw_imgs, (slice_resolution, new_height/height, new_width/width))
    zoomed_imgs = ((zoomed_imgs - np.min(zoomed_imgs)) / (np.max(zoomed_imgs) - np.min(zoomed_imgs))).astype(np.float32)
    np.save(os.path.join(des_home, x+".npy"), zoomed_imgs)

    zoomed_masks = zoom(raw_masks, (slice_resolution, new_height/height, new_width/width))
    zoomed_masks = ((zoomed_masks - np.min(zoomed_masks)) / (np.max(zoomed_masks) - np.min(zoomed_masks))).astype(np.float32)
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
