import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil

all_df = pd.read_csv('all_df.csv')

all_df
ptnt_lst = all_df['Patient ID'].drop_duplicates().to_list()

for p in ptnt_lst:
    # all_df.loc[]
    num = list(range(len(all_df.loc[all_df['Patient ID']==p])))
    num = [int(x) for x in num]
    all_df.loc[all_df['Patient ID']==p, 'slice_num'] = num

all_df.slice_num = all_df.slice_num.astype(int)

img_lst = all_df['File name'].to_list()
pid_lst = all_df['Patient ID'].to_list()
diag_lst = all_df['diag_num'].to_list()
slc_lst = all_df['slice_num'].to_list()

for (i, p, d, s) in zip(img_lst, pid_lst, diag_lst, slc_lst):
    if d == 0:
        clss = '/1NonCOVID/'
        print('non')
    elif d == 1:
        clss = '/2COVID/'
        print('cov')
    elif d == 2:
        clss = '/3CAP/'

    shutil.copy('archive/curated_data/curated_data' + clss + i, 'dataset3' + clss + p + '_' + str(s) + '.png')
