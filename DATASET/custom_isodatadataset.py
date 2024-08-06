
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

from skimage import io
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
import random
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte , img_as_float32
from skimage.filters import threshold_otsu , threshold_isodata , threshold_triangle , threshold_mean , threshold_minimum , threshold_li , threshold_yen
from skimage.color import rgb2gray
from skimage.morphology import erosion  , dilation , diamond , disk , binary_dilation , binary_erosion
from skimage.filters import threshold_minimum
from skimage.restoration import (
    calibrate_denoiser,
    denoise_wavelet,
    denoise_tv_chambolle,
    denoise_nl_means,
    estimate_sigma,
)




class customdataset(Dataset):
    def __init__(self , x,y , transform = None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.x.iloc[index] #dataframe path
        img = io.imread(img_path)
        img_r = resize(img , (100,100),anti_aliasing=True)/255.0
        img_r = rgb2gray(img_r)
        # img_r = equalize_adapthist(img_r)
        threshold = threshold_isodata(img_r , nbins = 5)
        binary_img_n = img_r > threshold
        binary_img_n = img_as_float32(img_r > threshold)
        
        binary_img_n = binary_img_n.astype(np.float32)
        binary_img_n = binary_img_n[np.newaxis,:,:]

        label = torch.tensor(int(self.y.iloc[index]))

        if self.transform:
            binary_img_n = self.transform(binary_img_n)

        return binary_img_n , label


    def __len__(self):
        return len(self.y)