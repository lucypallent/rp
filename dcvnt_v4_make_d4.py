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
import torch.nn as nn
import torch.nn.functional as F

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

###
# run = neptune.init(
#     project="lucypallent/research-project",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMmI3Y2EyOC1kZGMzLTRiNjgtYjY1MS04ZmZlMzA5MjJiYTYifQ==",
# )  # your credentials
############################ end of import

############################ start of classes / functions code for Unet
# removed above the bit which made teh dataset and txt files
# these can now be found in dataset3.py
# though still need to copy a file over from ImageSets/lung_test.txt to ImageSets-old/lung_test.txt

"""# Unet"""

# using the pretrained model originally

# CT volume needs to be 512x512 TxHxW


# from ops.dataset_ops import Train_Collatefn
# copied code from above file



def Train_Collatefn(data):
    all_F, all_L, all_info = [], [], []

    for i in range(len(data)):
        all_F.append(data[i][0])
        all_L.append(data[i][1])
        all_info.append(data[i][2])
    all_F = torch.cat(all_F, dim=0)
    all_L = torch.cat(all_L, dim=0)
    return all_F, all_L, all_info

# from dataset.dataset_test import CTDataset
# copied code from above file


# try:
#     from ops.dataset_ops import Rand_Transforms
# except:
#     #print ("Import external...")
#     import sys
#     sys.path.insert(0, "..")
#     from ops.dataset_ops import Rand_Transforms

readvdnames = lambda x: open(x).read().rstrip().split('\n')

class CTDataset(data.Dataset):
    def __init__(self, data_home="",
                       split="train",
                       sample_number=64,
                       clip_range=(0.2, 0.8),
                       logger=None):

        _meta_f = os.path.join(data_home, "ImageSets", "lung_{}.txt".format(split))
        # dataset3/NCOV-BF/ImageSets/lung_test.txt
        # Build a dictionary to record {path - label} pair
        meta    = [os.path.join(data_home, "NpyData", "{}.npy".format(x)) for x in readvdnames(_meta_f)]

        self.data_home = data_home
        self.split = split
        self.sample_number = sample_number
        self.meta = meta
        self.clip_range = (0.2, 0.7)
        #print (self.meta)
        self.data_len = len(self.meta)
        print ("[INFO] The true clip range is {}".format(self.clip_range))

    def __getitem__(self, index):
        data_path = self.meta[index]
        images = np.load(data_path)

        # CT clip
        num_frames = len(images)
        left, right = int(num_frames*self.clip_range[0]), int(num_frames*self.clip_range[1])
        images = images[left:right]

        num_frames = len(images)
        shape = images.shape
        #h, w = shape[1:]

        if False:
            from zqlib import imgs2vid
            imgs2vid(np.concatenate([images, masks*255], axis=2), "test.avi")
            import pdb
            pdb.set_trace()

        # To Tensor and Resize
        images = np.asarray(images, dtype=np.float32)
        images = images / 255.

        images = np.expand_dims(images, axis=1)          # Bx1xHxW, add channel dimension

        #info = {"name": data_path, "num_frames": num_frames, "shape": shape, "pad": ((lh,uh),(lw,uw))}
        info = {"name": data_path, "num_frames": num_frames, "shape": shape}

        th_img = torch.from_numpy(images.copy()).float()
        th_label = torch.zeros_like(th_img)

        return th_img, th_label, info

    def __len__(self):
        return self.data_len

    def debug(self, index):
        import cv2
        from zqlib import assemble_multiple_images
        th_img, th_label, info = self.__getitem__(index)
        # th_img: NxCxTxHxW

        img, label = th_img.numpy()[0, 0, :], th_label.numpy()[0]
        n, h, w = img.shape
        #if n % 2 != 0:
        #    img = np.concatenate([img, np.zeros((1, h, w))], axis=0)
        visual_img = assemble_multiple_images(img*255, number_width=16, pad_index=True)
        os.makedirs("debug", exist_ok=True)
        debug_f = os.path.join("debug/{}.jpg".format(\
                            info["name"].replace('/', '_').replace('.', '')))
        print ("[DEBUG] Writing to {}".format(debug_f))
        cv2.imwrite(debug_f, visual_img)


# if __name__ == "__main__":
#     # Read valid sliding: 550 seconds
#     ctd = CTDataset(data_home="dataset3/NCOV-BF", split="train", sample_number=4)
#     length = len(ctd)
#     ctd[10]

#     exit()
#     ctd.debug(0)
#     import time
#     s = time.time()
#     for i in range(length):
#         print (i)
#         th_img, th_label, info = ctd[i]
#     e = time.time()
#     print ("time: ", e-s)

#     #images, labels, info = ctd[0]
#     #for i in range(10):
#     #    ctd.debug(i)
#     import pdb
#     pdb.set_trace()





from importlib import import_module

# see above


random.seed(0); torch.manual_seed(0); np.random.seed(0)

CFG_FILE = "cfgs/test.yaml"

############### Set up Variables ###############
# with open(CFG_FILE, "r") as f: cfg = yaml.safe_load(f)

# MODEL_UID = 'unet'

# code to create the u-net model
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        print('here 1')
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        print('here 2')
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        print('here 3')
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        inter_channel = 16

        self.inc = DoubleConv(n_channels, inter_channel)
        self.down1 = Down(inter_channel, inter_channel*2)
        self.down2 = Down(inter_channel*2, inter_channel*4)
        self.down3 = Down(inter_channel*4, inter_channel*8)
        self.down4 = Down(inter_channel*8, inter_channel*8)
        self.up1 = Up(inter_channel*16, inter_channel*4, bilinear)
        self.up2 = Up(inter_channel*8, inter_channel*2, bilinear)
        self.up3 = Up(inter_channel*4, inter_channel, bilinear)
        self.up4 = Up(inter_channel*2, inter_channel, bilinear)
        self.outc = OutConv(inter_channel, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # 1/16
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



if __name__ == "__main__":
    unet = UNet(n_channels=1, n_classes=2)
    aa = torch.ones((2, 1, 128, 128))
    bb = unet(aa)
    print (bb.shape)
############################ end of classes / functions code for Unet
############################ start of making masks with Unet

PRETRAINED_MODEL_PATH = 'unet.pth' # "pretrained_model/unet-Epoch_00110-valid98.pth"
RESULE_HOME = 'unet-results'
NUM_WORKERS = 2
SAMPLE_NUMBER = -1 # All CT images
DATA_ROOT = 'dataset3/NCOV-BF' #'NCOV-BF/size368x368-dlmask'


Validset = CTDataset(data_home=DATA_ROOT,
                               split='test',
                               sample_number=SAMPLE_NUMBER)

# model = import_module(f"model.{MODEL_UID}")
# UNet = getattr(model, "UNet")

model = UNet(n_channels=1, n_classes=2)
model = torch.nn.DataParallel(model).cuda()

model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=f'cuda:{0}'), strict=True)

print('UNET IS LOADED')

# change so this is the whole dataset
ValidLoader = torch.utils.data.DataLoader(Validset,
                                    batch_size=1,
                                    num_workers=NUM_WORKERS,
                                    collate_fn=Train_Collatefn,
                                    shuffle=False,)

os.makedirs(RESULE_HOME, exist_ok=True)
os.makedirs("visual", exist_ok=True)

print('LEN OF VALIDLOADER')
print(len(ValidLoader))

with torch.no_grad():
    print('in loop')
    for i, (all_F, all_M, all_info) in enumerate(ValidLoader):
        print('enumerate loop')
        # print(i)
        all_E = []
        images = all_F.cuda()
        # print(images)
        #(lh, uh), (lw, uw) = all_info[0]["pad"]
        num = len(images)

        for ii in range(num):
            image = images[ii:ii+1]
            pred = model(image)
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            all_E.append(pred)

        all_E = torch.cat(all_E, dim=0).cpu().numpy().astype('uint8')
        all_OF2 = all_F[:, 0, :, :].cpu().numpy().astype('float32') * 255
        all_OF = np.uint8(all_F[:, 0, :, :].cpu().numpy().astype('float32') * 255)

        unique_id = all_info[0]["name"].split('/')[-1].replace('.npy', '')
        np.save("{}/{}.npy".format(RESULE_HOME, unique_id), all_OF)
        np.save("{}/{}-2.npy".format(RESULE_HOME, unique_id), all_OF2)

        np.save("{}/{}-dlmask.npy".format(RESULE_HOME, unique_id), all_E)

    print(i)
print('masks made')
############################ end of making masks with Unet
############################ start of functions for preprocessing .npys (creating d4)
### = just removed
import os
import numpy as np

from scipy.ndimage import zoom

readvdnames = lambda x: open(x).read().rstrip().split('\n')

src_home = 'unet-results'
des_home = 'dataset4/NCOV-BF/NpyData-size224x336'

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
    raw_imgs = np.load(os.path.join(src_home, x+"-2.npy")) # -2 is the img which appears like normal orig is blck sqr
    raw_masks = np.load(os.path.join(src_home, x+"-dlmask.npy"))
    length = len(raw_imgs)

    # clip_imgs = raw_imgs[int(length*clip_range[0]):int(length*clip_range[1])]
    ###
    # clip_masks = raw_masks[int(length*clip_range[0]):int(length*clip_range[1])]
    #
    # raw_imgs = clip_imgs
    # raw_masks = clip_masks
    ###
#####
    # zz, yy, xx = np.where(raw_masks)
    # cropbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])
    # crop_imgs = raw_imgs[cropbox[0, 0]:cropbox[0, 1],
    #                      cropbox[1, 0]:cropbox[1, 1],
    #                      cropbox[2, 0]:cropbox[2, 1]]
    #
    # crop_masks = raw_masks[cropbox[0, 0]:cropbox[0, 1],
    #                       cropbox[1, 0]:cropbox[1, 1],
    #                       cropbox[2, 0]:cropbox[2, 1]]
    #
    # raw_imgs = crop_imgs
    # raw_masks = crop_masks
#####
    height, width = raw_imgs.shape[1:3]
    zoomed_imgs = zoom(raw_imgs, (slice_resolution, new_height/height, new_width/width))
    np.save(os.path.join(des_home, x+".npy"), zoomed_imgs)
    zoomed_masks = zoom(raw_masks, (slice_resolution, new_height/height, new_width/width))
    np.save(os.path.join(des_home, x+"-dlmask.npy"), zoomed_masks)

    immasks = np.concatenate([zoomed_imgs, zoomed_masks*255], axis=2)[length//2]
    cv2.imwrite(f"debug/{x}.png", immasks)
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
