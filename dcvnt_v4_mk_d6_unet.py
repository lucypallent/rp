# uses the unet implemented masks found in unet-results vs running UNET

from zqlib import imgs2vid
import cv2
import os
import numpy as np
import skimage
from skimage import measure

readvdnames = lambda x: open(x).read().rstrip().split('\n')
pe_list = readvdnames(f"d6/image_sets/all_patients.txt")[::-1]

des_home = 'd6/unet' # '/content/test'
src_home = 'unet-results' # '/content' # where lung masks are saved 'unet-results'

def create_masked_lungs(x):
    print(x)
    raw_imgs = np.load(os.path.join(src_home, x+"-2.npy")) # -2 is the img which appears like normal orig is blck sqr
    raw_masks = np.load(os.path.join(src_home, x+"-dlmask.npy"))

    length = len(raw_imgs)

    raw_masked = np.zeros((length, 512, 512))
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

        # mask the images
        raw_masked[i] = cv2.bitwise_and(raw_imgs[i], raw_imgs[i], mask=dilated_slice)

    np.save(os.path.join(des_home, x+"-masked.npy"), raw_masked)
    np.save(os.path.join(des_home, x+"-dlmask.npy"), raw_masks)
    np.save(os.path.join(des_home, x+".npy"), raw_imgs)

############################ start of preprocessing .npys (creating d4)
from concurrent import futures

num_threads=10

with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    fs = [executor.submit(create_masked_lungs, x, ) for x in pe_list[::-1]]
    for i, f in enumerate(futures.as_completed(fs)):
        print ("{}/{} done...".format(i, len(fs)))
############################ end of preprocessing .npys (creating d4)
