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

###### start of making infnet form -2.npy in unet-results2

# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.
First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = COVIDDataset(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    return data_loader


class test_dataset2:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.npy')]
        readvdnames = lambda x: open(x).read().rstrip().split('\n')
        # self.pe_list = readvdnames(f"d6/image_sets/all_patients.txt")[::-1]
        # self.pe_list = readvdnames(f"dataset3/NCOV-BF/ImageSets-old/lung_test.txt")[::-1]
        self.pe_list = readvdnames(f"dataset4/NCOV-BF/ImageSets/lung_test.txt")[::-1]
        self.images = ['unet-results2/' + x + '-2.npy' for x in self.pe_list]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.test_transform = transforms.Compose([
                                                  transforms.ToTensor(),
                                                  # transforms.Resize([self.testsize, self.testsize]),

                                                  ])
        self.test_transform2 = transforms.Compose([
                                          transforms.ToPILImage(),
                                          # transforms.Resize(size = (self.testsize,self.testsize)),
                                          transforms.ToTensor(),
                                          # transforms.Normalize([0.5],[0.25]),
                                         ])
        # self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        npy = np.load(self.images[self.index])
        slc_num = len(npy)
        # image = self.rgb_loader(self.images[self.index])
        # ori_size = image.size
        npy = self.test_transform(npy) # torch.Size([352, 1, 64, 352])
        npy.unsqueeze_(0) # torch.Size([1, 512, 64, 512])
        npy = npy.permute(2, 0, 1,  3)
        # npy = npy.permute(1, 0, 2)

         # orch.Size([1, 512, 64, 512])

        npy = npy.repeat(1, 3, 1, 1)
        # print(npy.shape)
        # npy = self.test_transform2(npy)
        # print(npy.shape)
        t = transforms.ToPILImage()
        t2 = transforms.Resize(size = (self.testsize,self.testsize))
        t3 = transforms.ToTensor()
        t4 = transforms.Normalize([0.45],[0.25])
        t5 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ones used orig

        npy2 = torch.zeros(slc_num, 3, self.testsize, self.testsize)
        # torch.Size([3, 512, 64, 512])
        for i, n in enumerate(npy):
            n = t(n)
            n = t2(n)
            n = transforms.functional.adjust_brightness(n , brightness_factor=1.5)
            n = transforms.functional.adjust_contrast(n , contrast_factor=1.25)
            n = transforms.functional.rotate(n, 270)
            n = t3(n)
            n = torch.flip(n, (2,))

            # n = t4(n)
            n = t5(n)
            npy2[i] = n
            # print(n.shape)
        npy = npy2
        # print('new shape')
        # print(npy.shape)
        npy.unsqueeze_(0)
        # print(npy.shape)
        npy = npy.permute(1, 0, 2, 3, 4)
        # print(npy.shape)
        # print('done')

        # npy = self.test_transform2(npy.permute(2, 0, 1, 3)).unsqueeze(0).permute(1, 0, 2, 3, 4)
        # gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        # print(npy.shape)
        return npy, name

    def __len__(self):
        return self.size

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import cv2

__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model_lung_infection.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model_lung_infection pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model_lung_infection.
    Args:
        pretrained (bool): If True, returns a model_lung_infection pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model_lung_infection.
    Args:
        pretrained (bool): If True, returns a model_lung_infection pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        # Please replace it with your custom path
        model_state = torch.load('d6/mdls/res2net50_v1b_26w_4s-3cf99910.pth')
        model.load_state_dict(model_state)
        # model_lung_infection.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model_lung_infection.
    Args:
        pretrained (bool): If True, returns a model_lung_infection pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model_lung_infection.
    Args:
        pretrained (bool): If True, returns a model_lung_infection pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net152_v1b_26w_4s']))
    return model

# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.
First Version: Created on 2020-05-05 (@author: Ge-Peng Ji)
Second Version: Fix some bugs and edit some parameters on 2020-05-15. (@author: Ge-Peng Ji)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from Code.model_lung_infection.backbone.Res2Net import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, n_class):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, n_class, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class Inf_Net(nn.Module):
    def __init__(self, channel=32, n_class=1):
        super(Inf_Net, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        # ---- Partial Decoder ----
        self.ParDec = aggregation(channel, n_class)

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256+64, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, n_class, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64+64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64+64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)

        # ---- edge branch ----
        self.edge_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.edge_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv2d(64, n_class, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # ---- low-level features ----
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88

        # ---- high-level features ----
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        # ---- edge guidance ----
        x = self.edge_conv1(x1)
        x = self.edge_conv2(x)
        edge_guidance = self.edge_conv3(x)  # torch.Size([1, 64, 88, 88])
        lateral_edge = self.edge_conv4(edge_guidance)   # NOTES: Sup-2 (bs, 1, 88, 88) -> (bs, 1, 352, 352)
        lateral_edge = F.interpolate(lateral_edge,
                                     scale_factor=4,
                                     mode='bilinear')

        # ---- global guidance ----
        ra5_feat = self.ParDec(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat,
                                      scale_factor=8,
                                      mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1  # reverse
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = torch.cat((self.ra4_conv1(x), F.interpolate(edge_guidance, scale_factor=1/8, mode='bilinear')), dim=1)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4   # element-wise addition
        lateral_map_4 = F.interpolate(x,
                                      scale_factor=32,
                                      mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = torch.cat((self.ra3_conv1(x), F.interpolate(edge_guidance, scale_factor=1/4, mode='bilinear')), dim=1)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x,
                                      scale_factor=16,
                                      mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = torch.cat((self.ra2_conv1(x), F.interpolate(edge_guidance, scale_factor=1/2, mode='bilinear')), dim=1)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x,
                                      scale_factor=8,
                                      mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge


# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.
First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
# from scipy import misc
import imageio
import cv2

image_root = 'dataset3/NCOV-BF/NpyData/'
# testsize = 352
test_loader = test_dataset2(image_root, 512)
pth_path = 'd6/mdls/Inf-Net-100.pth'
save_path = 'dataset4/NCOV-BF/NpyData-imp10-infmask0010-test-1pc-redo/'
model = Inf_Net()
model.load_state_dict(torch.load(pth_path, map_location={'cuda:1':'cuda:0'}))
model.cuda()
model.eval()

################### commented out below to make other chnages
# transforms.functional.adjust_brightness
# transforms.functional.adjust_contrast
#
# kernel = np.ones((10,10), np.uint8)
# for i in range(test_loader.size):
#     images, name = test_loader.load_data()
#     images = images.cuda()
#     res2 = np.zeros((len(images), 512, 512))
#     res3 = np.zeros((len(images), 512, 512))
#     for i, img in enumerate(images): # ([64, 1, 3, 352, 352])
#         lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(img)
#         imgT = img[0].cpu()
#
#         res = lateral_map_2
#         res = res.sigmoid().data.cpu().numpy().squeeze()
#         res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#         res = cv2.resize(res, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
#         print(res)
#         res1 = (np.ceil(res-0.01)).astype(np.float32) # currently swapped the two lines
#
#         res = ((res - np.min(res)) / (np.max(res) - np.min(res))).astype(np.float32)
#
#
#         # # dilate the mask by 10 pixels
#         # dilated_res = cv2.dilate(res, kernel, iterations=1)
#
#         res2[i] = res1
#         res3[i] = res
#     print(res2.shape)
#     name = name.split('.')[0]
#     name = name[:-2]
#     name0 = name + '.npy'
#     np.save(os.path.join(save_path + name0), img.cpu())
#     name2 = name + '-infmask-orig.npy'
#     np.save(os.path.join(save_path + name2), res2)
#     # roundign up everything with an above than 0.1 chance of being an infection
# print('Test Done!')
################### commented out above to make other chnages
# add code from other versions to see if it works


# create the infnet limited to being the shape of the lungs
from zqlib import imgs2vid
import cv2
import os
import numpy as np
import skimage
from skimage import measure
from scipy import ndimage


readvdnames = lambda x: open(x).read().rstrip().split('\n')
pe_list = readvdnames(f"d6/image_sets/all_patients.txt")[::-1]

des_home = 'dataset4/NCOV-BF/NpyData-imp10-infmask0010-test-1pc-redo' # '/content/test'
src_home = 'unet-results2' # '/content' # where lung masks are saved 'unet-results2'

def create_masks(x):
    print(x)
    raw_imgs = np.load(os.path.join(des_home, x+'-infmask-orig.npy')) # -2 is the img which appears like normal orig is blck sqr
    raw_masks = np.load(os.path.join(src_home, x+'-dlmask.npy'))

    length = len(raw_imgs)

    raw_infmasked10 = np.zeros((length, 512, 512))
    raw_masked00 = np.zeros((length, 512, 512))

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
        dilated_slice10 = cv2.dilate(labels_slice, kernel, iterations=1)
        # fill in any holes in the lungs
        dilated_slice10 = ndimage.binary_fill_holes(dilated_slice10).astype(np.uint8)

        # dilate the lungs by 00 pixels
        dilated_slice00 = labels_slice #cv2.dilate(labels_slice, kernel, iterations=1)
        # fill in any holes in the lungs
        dilated_slice00 = ndimage.binary_fill_holes(dilated_slice00).astype(np.uint8)

        # dilate the covid lung infection selections by 10 pixels
        dilated_img = cv2.dilate(raw_imgs[i], kernel, iterations=1)

        # mask the images
        raw_infmasked10[i] = cv2.bitwise_and(dilated_img, dilated_img, mask=dilated_slice10)
        raw_masked00[i] = dilated_slice00

    # add together raw_masked00 and raw_infmasked10 to combine the any values of 2 become 1
    raw_imp_masked = raw_infmasked10 + raw_masked00
    raw_imp_masked[raw_imp_masked >= 1.0] = 1.0
    raw_imp_masked[raw_imp_masked < 1.0] = 0.0

    np.save(os.path.join(des_home, x+"-infmask.npy"), raw_infmasked10)
    np.save(os.path.join(des_home, x+"-dlmask.npy"), raw_masked00)
    # np.save(os.path.join(des_home, x+".npy"), raw_imgs)



create_masks('patient-P9')
print('==============================create masks works=================')

# ############################ start of preprocessing .npys (creating d4)
# from concurrent import futures
#
# num_threads=10
#
# with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
#     fs = [executor.submit(create_masks, x, ) for x in pe_list[::-1]]
#     for i, f in enumerate(futures.as_completed(fs)):
#         print ("{}/{} done...".format(i, len(fs)))
# ############################ end of preprocessing .npys (creating d4)

############################ create the masked lungs

src_home = 'unet-results2' # '/content' # where lung masks are saved 'unet-results2'
des_home = 'dataset4/NCOV-BF/NpyData-imp10-infmask0010-test-1pc-redo'

def create_masked_lungs(x):
    print(x)
    raw_imgs = np.load(os.path.join(src_home, x+"-2.npy")) # -2 is the img which appears like normal orig is blck sqr
    raw_masks = np.load(os.path.join(des_home, x+"-dlmask.npy")).astype('uint8')

    length = len(raw_imgs)

    raw_masked = np.zeros((length, 512, 512))
    kernel = np.ones((10,10), np.uint8)

    for i in range(length):
        # mask the images
        raw_masked[i] = cv2.bitwise_and(raw_imgs[i], raw_imgs[i], mask=raw_masks[i])

    raw_masked = ((raw_masked - np.min(raw_masked)) / (np.max(raw_masked) - np.min(raw_masked))).astype(np.float32)

    np.save(os.path.join(des_home, x+"-masked.npy"), raw_masked)

create_masked_lungs('patient-P9')
print('==============================create_masked_lungs works=================')


# ############################ start of preprocessing .npys (creating d4)
# from concurrent import futures
#
# num_threads=10
#
# with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
#     fs = [executor.submit(create_masked_lungs, x, ) for x in pe_list[::-1]]
#     for i, f in enumerate(futures.as_completed(fs)):
#         print ("{}/{} done...".format(i, len(fs)))
# ############################ end of preprocessing .npys (creating d4)

############################ start of functions for preprocessing .npys create 224x336
### = just removed
import os
import numpy as np

from scipy.ndimage import zoom

readvdnames = lambda x: open(x).read().rstrip().split('\n')

src_home = 'dataset4/NCOV-BF/NpyData-imp10-infmask0010-test-1pc-redo'
des_home = 'dataset4/NCOV-BF/NpyData-size224x336-imp10-infmask0010-test-1pc-redo'

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
    raw_masks = np.load(os.path.join(src_home, x+"-infmask.npy"))
    bin_masks = np.load(os.path.join(src_home, x+"-dlmask.npy"))
    bin_orig_inf_masks = np.load(os.path.join(src_home, x+"-infmask-orig.npy"))

    length = len(raw_imgs)

    height, width = raw_imgs.shape[1:3]
    zoomed_imgs = zoom(raw_imgs, (slice_resolution, new_height/height, new_width/width))
    zoomed_imgs = ((zoomed_imgs - np.min(zoomed_imgs)) / (np.max(zoomed_imgs) - np.min(zoomed_imgs))).astype(np.float32)
    np.save(os.path.join(des_home, x+".npy"), zoomed_imgs)

    zoomed_masks = zoom(raw_masks, (slice_resolution, new_height/height, new_width/width))
    zoomed_masks[zoomed_masks > 0.01] = 1.0
    zoomed_masks[zoomed_masks <= 0.01] = 0.0
    zoomed_masks = zoomed_masks.astype(np.float32)
    np.save(os.path.join(des_home, x+"-dlmask.npy"), zoomed_masks)

    zoomed_bin_masks = zoom(bin_masks, (slice_resolution, new_height/height, new_width/width))
    zoomed_bin_masks[zoomed_bin_masks > 0.01] = 1.0
    zoomed_bin_masks[zoomed_bin_masks <= 0.01] = 0.0
    zoomed_bin_masks = zoomed_bin_masks.astype(np.float32)
    np.save(os.path.join(des_home, x+"-dlmask-orig.npy"), zoomed_bin_masks)

    zoomed_bin_orig_inf_masks = zoom(bin_orig_inf_masks, (slice_resolution, new_height/height, new_width/width))
    zoomed_bin_orig_inf_masks[zoomed_bin_orig_inf_masks > 0.01] = 1.0
    zoomed_bin_orig_inf_masks[zoomed_bin_orig_inf_masks <= 0.01] = 0.0
    zoomed_bin_orig_inf_masks = zoomed_bin_orig_inf_masks.astype(np.float32)
    np.save(os.path.join(des_home, x+"-infmask-orig.npy"), zoomed_bin_orig_inf_masks)

    #imgs2vid(immasks, "debug/{}.avi".format(x))

resize_cta_images('patient-P9')
print('==============================resize_cta_images works=================')

############################ end of functions for preprocessing .npys (creating d4)

# ############################ start of preprocessing .npys (creating d4)
# from concurrent import futures
#
# num_threads=10
#
# with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
#     fs = [executor.submit(resize_cta_images, x, ) for x in pe_list[::-1]]
#     for i, f in enumerate(futures.as_completed(fs)):
#         print ("{}/{} done...".format(i, len(fs)))
# ############################ end of preprocessing .npys (creating d4)
