import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import random
import neptune.new as neptune

random.seed(0); torch.manual_seed(0); np.random.seed(0)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

run = neptune.init(
    project="lucypallent/natural-language-processing",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMmI3Y2EyOC1kZGMzLTRiNjgtYjY1MS04ZmZlMzA5MjJiYTYifQ==",
)  # your credentials

img = io.imread('dataset3/1NonCOVID/N314_16.png')
run.log_image('test_img', img)

from lungmask import mask

seg = mask.apply(img)
run.log_image('tes_seg', seg)
