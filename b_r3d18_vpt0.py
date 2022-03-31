# this is the version which uses the resnet3d model in the pytorch library

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
from skimage import io, transform
import os
import math
import numpy as np
from scipy import ndimage

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)
import numpy as np
np.random.seed(RANDOM_STATE)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Colab url commented out
# all_df = pd.read_csv('drive/MyDrive/curated_data/all_df.csv')
all_df = pd.read_csv('~/rp/dataset2/all_df.csv')


# CHANGE HERE FOR url in io.imread
# creating a class to read in this second dataset
# now adding in the custom dataset
class ctDataset2(Dataset):
    """CT Images dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df (dataframe): Dataframe previously read in from the csv file.
            # csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ct_frame = df
        self.root_dir = root_dir
        # self.transform = transform

        id_lst = list(self.ct_frame['Patient ID'].unique())
        self.id_df = pd.DataFrame(id_lst, columns = ['Patient ID'])
        self.id_df = self.id_df.merge(self.ct_frame[['Patient ID', 'Diagnosis', 'diag_num']], how='left', on='Patient ID').drop_duplicates(ignore_index=True)

        # currently not normalising cause causes issues
        self.transform = transforms.Compose([transforms.ToTensor(),
                                ])

    def __len__(self):
        return len(self.id_df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) == int:
            idx = [idx]

        patient_id = self.id_df.iloc[idx, 0].tolist()
        img_class = self.id_df.iloc[idx, 1].tolist()
        img_class_num = torch.tensor(self.id_df.iloc[idx, 2].values)[0]

        for pi, ic in zip(patient_id, img_class):

            img1 = np.zeros((1, 512, 512, 3))
            if ic == 'COVID-19':
                clss = '/2COVID/'
            elif ic == 'Normal':
                clss = '/1NonCOVID/'
            elif ic == 'CAP':
                clss = '/3CAP/'

            imgs = torch.empty((1, 128, 128, 3)).to(device)
            for i in range(64):
                # Colab url commented out
                # img = torch.from_numpy(io.imread('/content/drive/MyDrive/curated_data/dataset2' + clss + pi + '_' + str(i) + '.png')).to(device)
                img = torch.from_numpy(io.imread('~/rp/dataset2/dataset2' + clss + pi + '_' + str(i) + '.png')).to(device)

                imgs = torch.cat((imgs, img.reshape(1, 128, 128, 3)), 0)
            imgs = imgs[1:].permute(3, 0, 1, 2) / 255

        sample = {'images': imgs, 'classes': img_class, 'pIDs': patient_id, 'labels': img_class_num}
        return sample


from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

all_patient_df = all_df.drop(['File name', 'Slice'], axis=1).drop_duplicates()
# all_patient_df

all_patient_df_train, all_patient_df_test = train_test_split(all_patient_df, train_size =0.8, random_state=RANDOM_STATE, stratify=all_patient_df['Diagnosis'])

all_df_train = all_patient_df_train.merge(all_df[['File name', 'Patient ID', 'Slice']], on='Patient ID')
all_df_train = all_df_train[['File name', 'Patient ID', 'Slice', 'Diagnosis', 'diag_num']]
all_df_train.sort_values(['Patient ID', 'Slice'], ascending=True, inplace=True)

all_df_test = all_patient_df_test.merge(all_df[['File name', 'Patient ID', 'Slice']], on='Patient ID')
all_df_test = all_df_test[['File name', 'Patient ID', 'Slice', 'Diagnosis', 'diag_num']]
all_df_test.sort_values(['Patient ID', 'Slice'], ascending=True, inplace=True)

# Colab url commented out
# ctTrainDataset = ctDataset2(df=all_df_train, root_dir='/content/drive/MyDrive/curated_data/curated_data')
ctTrainDataset = ctDataset2(df=all_df_train, root_dir='~/rp/dataset2/dataset2')
# ctTestDataset = ctDataset2(df=all_df_test, root_dir='/content/drive/MyDrive/curated_data/curated_data')
ctTestDataset = ctDataset2(df=all_df_test, root_dir='~/rp/dataset2/dataset2')


import torchvision.models as models
r3d_18 = models.video.r3d_18(pretrained = True).to(device)

# freeze all the network except the final layer, so gradients are
# not computed in backward() - to do FIXED FEATURE EXTRACTION
# frome here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
for p in r3d_18.parameters():
    p.requires_grad = False

# changing final 2 layers of r3d_18
# r3d_18.avgpool = torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
r3d_18.fc = torch.nn.Linear(r3d_18.fc.in_features, 3).to(device)

BATCH_SIZE = 4

ctTrainDataloader=torch.utils.data.DataLoader(ctTrainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
ctTestDataloader=torch.utils.data.DataLoader(ctTestDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

NUM_EPOCHS = 100
# BEST_MODEL_PATH = 'best_model.pth'

optimizer = optim.SGD(r3d_18.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(ctTrainDataloader):
        images = data['images'].to(device)
        labels = data['labels'].to(device)
        optimizer.zero_grad()
        outputs = r3d_18(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

true = torch.tensor([]).to(device)
pred = torch.tensor([]).to(device)

 # test for t
for i, data in enumerate(ctTestDataloader):

    images = data['images'].to(device)
    labels = data['labels'].to(device)

    outputs = r3d_18(images)
    true = torch.cat((true, labels), 0)


    # Get predictions from the maximum value
    predicted = torch.max(outputs.data, 1)[1]
    pred = torch.cat((pred, predicted), 0)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

true = true.to('cpu')
pred = pred.to('cpu')

target_names = ['NonCOVID', 'COVID', 'CAP']

# get precision, recall, f1-score
print(classification_report(true, pred, target_names=target_names, digits=4))

# get accuracy each individual class
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(true, pred)
print(matrix.diagonal()/matrix.sum(axis=1))

# get accuracy averaged out across class
print(accuracy_score(true, pred))

# Colab path commented out
# torch.save(r3d_18.state_dict(), 'drive/MyDrive/curated_data/baseline_r3d18_v0_test_2epoch.pth')
torch.save(r3d_18.state_dict(), 'b_r3d18_v0.pth')
