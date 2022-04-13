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
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from lungmask import mask
import SimpleITK as sitk
import skimage
from skimage import io
import numpy as np
import pydicom

from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *
from fastai.medical.imaging import *


pth = 'COVID19_0001/COVID19-0101.dcm' # path is correct
cpth = 'COVID19_0001'
pth2 = '62448201'
pth3 = 'unet-results/patient-P7-2.npy'

ds = pydicom.dcmread(pth)#[0x7fe0, 0x0010].value
ds.decompress()
print(ds.pixel_array.min())
print(ds.pixel_array.max())
# ds = np.clip(ds.pixel_array,-1024,600) # may actually start at 0 ngl
ds = np.clip(ds.pixel_array,0,1624) # may actually start at 0 ngl
amin = 0
amax = 1624
ds = (ds - amin) / (amax + amin)
print(ds.pixel_array.min())
print(ds.pixel_array.max())
print('decompressed')

citems = get_dicom_files(cpth)
patient2 = dcmread(citems[0])

ds = pydicom.dcmread(pth)
print(ds)
print(ds.pixel_array)
print('WORKS')
#
# # i think i need to normalise This
# # The HU values range between  and , with higher values being obtained from bones and metal implants in the body and lung regions typically ranging in . Similar to literature, we process chest CT scans such that values higher than  are mapped to  and the range  is normalized to the  linearly.
#
# # also 256 x 256
#
# input_image = sitk.ReadImage(pth)
# print(type(input_image))
# input_image = sitk.ReadImage(pth2)
# print(type(input_image))
# model = mask.get_model('unet','R231CovidWeb')
# result = mask.apply(input_image, model)
#
# result = mask.apply(np.load(pth3)*255, model)
#
#
# print(type(result))
# print(result.shape)
# print(result.max())
# print(result.min())
# print('WORKS!')
#
# # dataset.pixel_array - how ot get pixel info ie below would display imgs
# # plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
# # plt.show()
#
#
# # ####### start of dicom run
# # # input_image = sitk.ReadImage(pth5)
# # #
# # # model = mask.get_model('unet','R231CovidWeb')
# # # result = mask.apply(input_image, model)#, noHU=True)
# # # result = (result/(result.max())*255).astype(np.uint8)
# # #
# # # print(type(result))
# # # print(result.shape)
# # # print(result.max())
# # # print(result.min())
# # # print('WORKS!')
# # #
# # # io.imsave(pth6, result[0])
# # ####### end of dicom run
# # #
# # # read in image concat save as png read in the png with sitk
# # # this should then work
# # # make rgb
# # img = io.imread(pth2).reshape((512, 512, 1))
# # # print(img.shape)
# #
# # img0 = np.empty((512, 512, 1))
# # img1 = np.concatenate((img0, img, img, img), axis=2)[:,:,1:] / 255
# # print(img1.shape)
# #
# # img1 = (img1/(img1.max())*255).astype(np.uint8)
# # io.imsave(pth3, img1)
# #
# # # up brightness
# # from PIL import Image, ImageEnhance
# # im = Image.open(pth3)
# # enhancer = ImageEnhance.Brightness(im)
# #
# # factor = 2 #increase contrast
# # im_output = enhancer.enhance(factor)
# # plt.imshow(im_output)
# # im_output.save(pth3)
# #
# # input_image = sitk.ReadImage(pth8)
# #
# # print('.png image')
# # print(type(input_image))
# # img_arr = sitk.GetArrayFromImage(input_image)
# # print(img_arr.shape)
# # print(img_arr)
# # print(img_arr.max())
# # print(img_arr.min())
# #
# # print('.dicom image')
# # input_image2 = sitk.ReadImage(pth7)
# # print(type(input_image2))
# # img_arr2 = sitk.GetArrayFromImage(input_image2)
# # print(img_arr2)
# # print(img_arr2.max())
# # print(img_arr2.min())
# #
# # # input_image = sitk.ReadImage(pth)
# #
# # # print(input_image.max())
# # # print(input_image.min())
# # # print(len(input_image))
# # # print(type(input_image))
# #
# # # print('2nd IMAGE')
# # # input_image2 = sitk.ReadImage(pth2)
# # # print(input_image2)
# # # print(len(input_image2))
# # # print(type(input_image2))
# #
# #
# # # print(input_image.shape)
# # # input_image = skimage.color.gray2rgb(input_image)
# # model = mask.get_model('unet','R231CovidWeb')
# # result = mask.apply(input_image, model, noHU=True)
# #
# # print(result.max())
# # print(result.min())
# #
# # result = result * 255
# #
# # print(result.max())
# # print(result.min())
# #
# # # result = (result/(result.max())*255).astype(np.uint8)
# # result = result[0]
# #
# # io.imsave(pth4, result)
