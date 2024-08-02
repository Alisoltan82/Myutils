import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
import os
from PIL import Image
import nibabel as nib
import pydicom as dicom
import nibabel as nib
from glob import glob
import random
import cv2
from pathlib import Path
from tqdm.auto import tqdm
sns.set(style='whitegrid')



def compare(image1, image2 , title = str):
    '''
    takes 2 images an put them side by side
    '''
    fig,axs = plt.subplots(1,2,figsize = (9,8))
    axs[0].imshow(image1,cmap = plt.cm.gray)
    axs[0].set_title(f'original')

    axs[1].imshow(image2 , cmap = plt.cm.gray)
    axs[1].set_title(f'{title}')



def dataset_check(dataset , count = int):
    if not dataset or count <= 0:
     return
    
    for i in range(count):
        fig, axs = plt.subplots(count, figsize=(10, 6 * count))
        axs = axs.flatten()  # Make a flat list of axes

        for i in range(min(count, len(dataset))):
            img, label = random.choice(dataset)
            axs[i].imshow(img)
            axs[i].set_title(label)
            axs[i].axis('off')

        plt.show()



def dataloader_check(dataloader , count = int):
    """
    Check images in a data loader and make subplots grid-based on the count input.

    Args:
        dataloader (torch.utils.data.DataLoader): Data loader object.
        count (int, optional): Number of subplots to display. Defaults to 1.

    Returns:
        None
    """
    if not dataloader or count <= 0:
        return

    dataset = list(dataloader.dataset)
    random.shuffle(dataset)

    fig, axs = plt.subplots(count, figsize=(10, 6 * count))
    axs = axs.flatten()  # Make a flat list of axes

    for i in range(min(count, len(dataset))):
        img, label = dataset[i]
        axs[i].imshow(img.permute(1, 2, 0))  # Assuming image is stored as (C, H, W)
        axs[i].set_title(label)
        axs[i].axis('off')

    plt.show()



def plot_images_per_class(df, target_column, img_path_column, count):
    """
    Plot images for each class in the dataframe, limited to the specified count.

    Args:
        df (pandas.DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        img_path_column (str): Name of the image path column.
        count (int): Maximum number of samples to plot per class.

    Returns:
        None
    """
    # Get unique classes and their counts
    classes = df[target_column].unique()
    class_counts = {cls: len(df[df[target_column] == cls]) for cls in classes}

    # Create a figure with subplots for each class
    fig, axs = plt.subplots(len(classes), figsize=(10, 6 * len(classes)))

    # Loop through each class and plot its images
    for i, cls in enumerate(classes):
        img_paths = df[df[target_column] == cls][img_path_column].tolist()
        img_files = [os.path.join(os.getcwd(), path) for path in img_paths]
        axs[i].imshow(plt.imread(img_files[0]))
        axs[i].set_title(f"{cls} ({class_counts[cls]} samples)")
        axs[i].axis('off')

    # Limit the number of images per class
    for i, cls in enumerate(classes):
        if len(img_files) > count:
            axs[i].imshow(plt.imread(img_files[count - 1]))
            axs[i].set_title(f"{cls}  ({count} samples)")
            axs[i].axis('off')

     # Show the plot
    plt.show()
                                


def enhancing_binary_gray_images(image1, image2, print=True):
    """
    Enhance binary gray images and display original and enhanced images side by side.

    Args:
        image1 (str or numpy.ndarray): Path to the first image file or a 2D/3D numpy array.
        image2 (str or numpy.ndarray): Path to the second image file or a 2D/3D numpy array.
        print (bool, optional): Whether to print image dimensions and statistics. Defaults to True.

    Returns:
        None
    """

    # Load images
    if isinstance(image1, str):
        if image1.endswith('.nii') or image1.endswith('.nif'):
            image1 = nib.load(image1).get_fdata()
        elif image1.endswith('.dcm'):
            image1 = dc.read_file(image1)[0].pixel_array
        else:
            image1 = Image.open(image1).convert('L')
    if isinstance(image2, str):
        if image2.endswith('.nii') or image2.endswith('.nif'):
            image2 = nib.load(image2).get_fdata()
        elif image2.endswith('.dcm'):
            image2 = dicom.read_file(image2)[0].pixel_array
        else:
            image2 = Image.open(image2).convert('L')

    # Print image dimensions and statistics
    if print:
        print(f"Image 1 dimensions: {image1.shape}")
        print(f"Image 1 max value: {np.max(image1)}")
        print(f"Image 1 min value: {np.min(image1)}")
        print(f"Image 1 mean value: {np.mean(image1)}")

    # Apply enhancing transformations
    enhanced_image1 = np.clip(image1, 0, 255)
    enhanced_image2 = np.clip(image2, 0, 255)

    # Display original and enhanced images side by side
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(image1, cmap='gray')
    axs[0, 0].set_title('Original Image 1')
    axs[0, 1].imshow(enhanced_image1, cmap='gray')
    axs[0, 1].set_title('Enhanced Image 1')
    axs[1, 0].imshow(image2, cmap='gray')
    axs[1, 0].set_title('Original Image 2')
    axs[1, 1].imshow(enhanced_image2, cmap='gray')
    axs[1, 1].set_title('Enhanced Image 2')

    plt.show()
    

def df_split_Norm_gray(img_dir=str, df=pd.DataFrame, img_path_col=str, label_name_col=str, size:tuple=(224,224), save_dir=str):
    sums, sums_squared = 0, 0

    for c, patient in enumerate(tqdm(df)):
        img_path = df.iloc[c][img_path_col]
        
        # Check if the image is a DICOM file
        if img_path.endswith('.dcm'):
            ds = dicom.dcmread(img_path)
            img = ds.pixel_array / 255.0
            
        # Check if the image is a NIFTI file
        elif img_path.endswith('.nii.gz') or img_path.endswith('.nii'):
            img = nib(img_path).get_data()
            
        # If it's not a DICOM or NIFTI file, assume it's a grayscale image
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        img_r = cv2.resize(img, size).astype(np.float16)
        
        label = df.iloc[c][label_name_col]
        
        train_folder = 'train'
        save_path = os.path.join(save_dir, train_folder, str(label))
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f'{c:03d}'), img_r)
        
        normalizer = size[0] * size[1]
        sums += np.sum(img_r) / normalizer
        sums_squared += (img_r ** 2).sum() / normalizer

    mean = sums / len(df)
    print(f'train images mean after resize = {mean}')

    std = np.sqrt((sums_squared/len(df)) - mean**2)
    print(f'train images standard deviation after resize = {std}')