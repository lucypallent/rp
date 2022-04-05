import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil

all_df = pd.read_csv('dataset3/all_df.csv')

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

    shutil.copy('curated_data/curated_data' + clss + i, 'dataset3' + clss + p + '_' + str(s) + '.png')

# create .npy files
all_df['counts'] = all_df.groupby(['Patient ID'])['Diagnosis'].transform('count')
c_p_lst = all_df.loc[all_df['diag_num']==1, ['Patient ID', 'counts']].drop_duplicates()['Patient ID'].to_list()
c_p_cnt = all_df.loc[all_df['diag_num']==1, ['Patient ID', 'counts']].drop_duplicates()['counts'].to_list()

n_p_lst = all_df.loc[all_df['diag_num']==0, ['Patient ID', 'counts']].drop_duplicates()['Patient ID'].to_list()
n_p_cnt = all_df.loc[all_df['diag_num']==0, ['Patient ID', 'counts']].drop_duplicates()['counts'].to_list()

p_p_lst = all_df.loc[all_df['diag_num']==2, ['Patient ID', 'counts']].drop_duplicates()['Patient ID'].to_list()
p_p_cnt = all_df.loc[all_df['diag_num']==2, ['Patient ID', 'counts']].drop_duplicates()['counts'].to_list()

device = 'cpu'
clss = '/1NonCOVID/'
for (pi, cnt) in zip(n_p_lst, n_p_cnt):
    imgs = torch.empty((1, 512, 512)).to(device)
    for i in range(cnt):
        img = torch.from_numpy(io.imread('dataset3' + clss + pi + '_' + str(i) + '.png')).to(device)

        imgs = torch.cat((imgs, img), 0)
    imgs = imgs[1:] / 255 # creates a tensor (64, 512, 512) which is TxHxW

    np.save('dataset3/NCOV-BF/NpyData/patient-' + pi + '.npy', imgs.to('cpu').numpy())

clss = '/2COVID/'
for (pi, cnt) in zip(c_p_lst, c_p_cnt):
    imgs = torch.empty((1, 512, 512)).to(device)
    for i in range(cnt):
        img = torch.from_numpy(io.imread('dataset3' + clss + pi + '_' + str(i) + '.png')).to(device)

        imgs = torch.cat((imgs, img), 0)
    imgs = imgs[1:] / 255 # creates a tensor (64, 512, 512) which is TxHxW

    np.save('dataset3/NCOV-BF/NpyData/patient-' + pi + '.npy', imgs.to('cpu').numpy())

clss = '/3CAP/'
for (pi, cnt) in zip(p_p_lst, p_p_cnt):
    imgs = torch.empty((1, 512, 512)).to(device)
    for i in range(cnt):
        img = torch.from_numpy(io.imread('dataset3' + clss + pi + '_' + str(i) + '.png')).to(device)

        imgs = torch.cat((imgs, img), 0)
    imgs = imgs[1:] / 255 # creates a tensor (64, 512, 512) which is TxHxW

    np.save('dataset3/NCOV-BF/NpyData/patient-' + pi + '.npy', imgs.to('cpu').numpy())

#write the .txt files
path_to_file = 'dataset3/NCOV-BF/ImageSets/lung_test.txt'

lines = []
lines.extend(N_lst)
lines.extend(C_lst)
# lines.extend(P_lst)


f = open(path_to_file,"w")

for line in lines:
    # need to add 'patient-' to each
    f.write('patient-' + line)
    f.write('\n')
f.close()

random.shuffle(N_lst)
random.shuffle(C_lst)
random.shuffle(P_lst)

N_lst_train = N_lst[:198]
N_lst_valid = N_lst[198:198+66]
N_lst_test = N_lst[198+66:]

C_lst_train = C_lst[:81]
C_lst_valid = C_lst[81:81+27]
C_lst_test = C_lst[81+27:]

P_lst_train = P_lst[:32]
P_lst_valid = P_lst[32:32+11]
P_lst_test = P_lst[32+11:]

def write_txt(path, lst):
  lines = []
  lines.extend(lst)

  f = open(path,"w")

  for line in lines:
      # need to add 'patient-' to each
      f.write('patient-' + line)
      f.write('\n')
  f.close()

write_txt('dataset3/NCOV-BF/ImageSets/normal_train.txt', N_lst_train)
write_txt('dataset3/NCOV-BF/ImageSets/normal_test.txt', N_lst_test)
write_txt('dataset3/NCOV-BF/ImageSets/normal_valid.txt', N_lst_valid)

write_txt('dataset3/NCOV-BF/ImageSets/ncov_train.txt', C_lst_train)
write_txt('dataset3/NCOV-BF/ImageSets/ncov_test.txt', C_lst_test)
write_txt('dataset3/NCOV-BF/ImageSets/ncov_valid.txt', C_lst_valid)

write_txt('dataset3/NCOV-BF/ImageSets/cap_train.txt', P_lst_train)
write_txt('dataset3/NCOV-BF/ImageSets/cap_test.txt', P_lst_test)
write_txt('dataset3/NCOV-BF/ImageSets/cap_valid.txt', P_lst_valid)

print('DONE AND WORKS')
