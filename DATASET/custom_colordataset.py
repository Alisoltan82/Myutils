
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


if __name__ == "__main__" :

    class customdataset(Dataset):
        def __init__(self , x,y , transform = None):
            self.x = x
            self.y = y
            self.transform = transform

        def __getitem__(self, index):
            img_path = self.x.iloc[index] #dataframe path
            img = io.imread(img_path)
            img = resize(img , (100,100),anti_aliasing=True)
            # img_r = rgb2gray(img)
            
            threshold = threshold_isodata(img , nbins = 3)
            binary_img_n = img > threshold
            binary_img_n = img_as_float32(img > threshold)
            binary_img_n = binary_erosion(binary_img_n)
            binary_img_n = binary_dilation(binary_img_n)

            
            img = img.transpose(2,0,1)
            
            binary_img_n = binary_img_n.transpose(2,0,1)
            
            
            merged_img = np.hstack((img, binary_img_n))
            
            

            label = torch.tensor(int(self.y.iloc[index]))

            if self.transform:
                merged_img = self.transform(merged_img)

            return merged_img , label


        def __len__(self):
            return len(self.y)