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

readvdnames = lambda x: open(x).read().rstrip().split('\n')
pe_list = readvdnames(f"dataset3/NCOV-BF/ImageSets-old/lung_test.txt")[::-1]

# where saving
des_home = 'dataset4/NCOV-BF/NpyData-size224x336-skip'

# where loading from
src_home = 'dataset4/NCOV-BF/NpyData-size224x336'

def skip_image_slices(x):        # dtype is "PE"/"NORMAL"
    print (x)
    if os.path.isfile(os.path.join(des_home, x+".npy")) is True:
        return
    ### raw_imgs = np.uint8(np.load(os.path.join(src_home, x+".npy")))
    raw_imgs = np.load(os.path.join(src_home, x+".npy"))
    raw_masks = np.load(os.path.join(src_home, x+"-dlmask.npy"))

    skipped_imgs = raw_imgs[::2]
    skipped_masks = raw_masks[::2]

    np.save(os.path.join(des_home, x+".npy"), skipped_imgs)
    np.save(os.path.join(des_home, x+"-dlmask.npy"), skipped_masks)

from concurrent import futures

num_threads=10

with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(skip_image_slices, x, ) for x in pe_list[::-1]]
    for i, f in enumerate(futures.as_completed(fs)):
        print ("{}/{} done...".format(i, len(fs)))
############################ end of preprocessing .npys (creating d4)
