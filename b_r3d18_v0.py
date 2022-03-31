# generate r3d18 based on here https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


# start of code used to run the below
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


# using this model here
# r3d18_K_200ep.pth

# --model resnet --model_depth 34 --n_pretrain_classes 700

r3d18 = generate_model(18, n_classes=700)
PATH = 'r3d18_K_200ep.pth'
# PATH = '/content/drive/MyDrive/curated_data/r3d18_K_200ep.pth'

# Model class must be defined somewhere
r3d18.load_state_dict(torch.load(PATH)['state_dict'])
r3d18.eval()
r3d18 = r3d18.to(device)

# so we're doing feature extraction
for p in r3d18.parameters():
    p.requires_grad = False

# change to classify for 3 features
r3d18.fc = nn.Linear(r3d18.fc.in_features, 3).to(device) # 3 classes


BATCH_SIZE = 4

ctTrainDataloader=torch.utils.data.DataLoader(ctTrainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
ctTestDataloader=torch.utils.data.DataLoader(ctTestDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

NUM_EPOCHS = 100
# BEST_MODEL_PATH = 'best_model.pth'

optimizer = optim.SGD(r3d18.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(ctTrainDataloader):
        images = data['images'].to(device)
        labels = data['labels'].to(device)
        optimizer.zero_grad()
        outputs = r3d18(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

true = torch.tensor([]).to(device)
pred = torch.tensor([]).to(device)

 # test for t
for i, data in enumerate(ctTestDataloader):

    images = data['images'].to(device)
    labels = data['labels'].to(device)

    outputs = r3d18(images)
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
# torch.save(r3d18.state_dict(), 'drive/MyDrive/curated_data/baseline_r3d18_v0_test_2epoch.pth')
torch.save(r3d_34.state_dict(), 'b_r3d18_v0.pth')
