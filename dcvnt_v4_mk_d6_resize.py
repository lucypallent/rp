import os
import numpy as np
from scipy.ndimage import zoom
from zqlib import imgs2vid
import cv2

# src_home = '/content' #'unet-results'
inf_home = 'd6/infnet'
unt_home = 'd6/unet'
des_home = 'd6/resized224x336' #'dataset4/NCOV-BF/NpyData-size224x336'

os.makedirs(des_home, exist_ok=True)

new_size = (224, 336)   # 224x336       # average isotropic shape: 193x281
new_height, new_width = new_size
clip_range = (0.15, 1)
slice_resolution = 1

readvdnames = lambda x: open(x).read().rstrip().split('\n')
pe_list = readvdnames(f"d6/image_sets/all_patients.txt")[::-1]

def resize_images(x):        # dtype is "PE"/"NORMAL"
    print(x)
    raw_npy = np.load(os.path.join(unt_home, x+".npy")) # original ct scan
    raw_dlm = np.load(os.path.join(unt_home, x+"-dlmask10.npy")) # lungs bin mask
    raw_msk = np.load(os.path.join(unt_home, x+"-masked.npy")) # masked lungs
    raw_inf = np.load(os.path.join(inf_home, x+"-infmask.npy")) # infnet bin mask
    raw_infmsk = np.load(os.path.join(inf_home, x+"-inf10masked10.npy")) # masked infnet bin mask

    print('raw')
    length = len(raw_npy)

    clip_npy = raw_npy[int(length*clip_range[0]):int(length*clip_range[1])]
    clip_dlm = raw_dlm[int(length*clip_range[0]):int(length*clip_range[1])]
    clip_msk = raw_msk[int(length*clip_range[0]):int(length*clip_range[1])]
    clip_inf = raw_inf[int(length*clip_range[0]):int(length*clip_range[1])]
    clip_infmsk = raw_infmsk[int(length*clip_range[0]):int(length*clip_range[1])]
    print('clipped')
    raw_npy = clip_npy
    raw_dlm = clip_dlm
    raw_msk= clip_msk
    raw_inf = clip_inf
    raw_infmsk = clip_infmsk

    zz, yy, xx = np.where(raw_dlm)
    cropbox = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])
    crop_npy = raw_npy[cropbox[0, 0]:cropbox[0, 1],
                         cropbox[1, 0]:cropbox[1, 1],
                         cropbox[2, 0]:cropbox[2, 1]]
    crop_dlm = raw_dlm[cropbox[0, 0]:cropbox[0, 1],
                         cropbox[1, 0]:cropbox[1, 1],
                         cropbox[2, 0]:cropbox[2, 1]]
    crop_msk = raw_msk[cropbox[0, 0]:cropbox[0, 1],
                          cropbox[1, 0]:cropbox[1, 1],
                          cropbox[2, 0]:cropbox[2, 1]]
    crop_inf = raw_inf[cropbox[0, 0]:cropbox[0, 1],
                         cropbox[1, 0]:cropbox[1, 1],
                         cropbox[2, 0]:cropbox[2, 1]]
    crop_infmsk = raw_infmsk[cropbox[0, 0]:cropbox[0, 1],
                         cropbox[1, 0]:cropbox[1, 1],
                         cropbox[2, 0]:cropbox[2, 1]]
    print('cropped')

    raw_npy = crop_npy
    raw_dlm = crop_dlm
    raw_msk = crop_msk
    raw_inf = crop_inf
    raw_infmsk = crop_infmsk

    print('save')
    height, width = crop_infmsk.shape[1:3]
    zoomed_npy = zoom(raw_npy, (slice_resolution, new_height/height, new_width/width))
    print(zoomed_npy.shape)
    np.save(os.path.join(des_home, x+'.npy'), zoomed_npy)
    zoomed_dlm = zoom(raw_dlm, (slice_resolution, new_height/height, new_width/width))
    print(zoomed_dlm.shape)
    np.save(os.path.join(des_home, x+'-dlmask10.npy'), zoomed_dlm)
    zoomed_msk = zoom(raw_msk, (slice_resolution, new_height/height, new_width/width))
    print(zoomed_msk.shape)
    np.save(os.path.join(des_home, x+'-masked.npy'), zoomed_msk)
    zoomed_inf = zoom(raw_inf, (slice_resolution, new_height/height, new_width/width))
    print(zoomed_inf.shape)
    np.save(os.path.join(des_home, x+'-infmask.npy'), zoomed_inf)
    zoomed_infmsk = zoom(raw_infmsk, (slice_resolution, new_height/height, new_width/width))
    print(zoomed_infmsk.shape)
    np.save(os.path.join(des_home, x+'-inf10masked10k.npy'), zoomed_infmsk)
############################ end of functions for preprocessing .npys (creating d4)

############################ start of preprocessing .npys (creating d4)
from concurrent import futures

num_threads=10

with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(resize_images, x, ) for x in pe_list[::-1]]
    for i, f in enumerate(futures.as_completed(fs)):
        print ("{}/{} done...".format(i, len(fs)))
############################ end of preprocessing .npys (creating d4)
