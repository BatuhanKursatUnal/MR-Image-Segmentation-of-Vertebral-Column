#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:28:15 2024

@author: batuhanmac
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas
import importlib
import SimpleITK as sitk
from ipywidgets import interact
import re

# Necessary packages that are used in this project are listed below. 
#The following code, which is taken from simpleitk tutorials directly, checks that all packages are installed
required_packages = {
    "numpy",
    "matplotlib",
    "ipywidgets",
    "scipy",
    "pandas",
    "numba",
    "multiprocess",
    "SimpleITK",
}

problem_packages = list()

for package in required_packages:
    try:
        p = importlib.import_module(package)
    except ImportError:
        problem_packages.append(package)

if len(problem_packages) == 0:
    print("All is well.")
else:
    print(
        "The following packages are required but not installed: "
        + ", ".join(problem_packages)
    )


# Setting up the chosen image viewer, i.e. 3DSlicer
#Default app is ImageJ, another commonly used one is Fiji
image_viewer = sitk.ImageViewer()
image_viewer.SetApplication("/pathtoimageviewerapp/Slicer") #Remember to adjust it according to your local path


# Retrieve the information about image volumes
mr_volume_dir = ("/pathtodataset/images") #Remember to adjust it according to your local path
mr_images = os.listdir(mr_volume_dir)
size_list = []
mr_imagefiles = []

## To get rid of the redundant data
for file in mr_images:
    if file[0].isnumeric():
        mr_imagefiles += [file]
        
## Sorting out the lists to have a ordered size list
### Extracting the numeric part from the filename
def extract_number(mri_files):
    match = re.match(r"(\d+)_", mri_files)
    if match:
        return int(match.group(1)) 
    else: 
        float('inf')

sorted_mr_imagefiles = sorted(mr_imagefiles, key=extract_number)
    
for files in sorted_mr_imagefiles:    
    mr_images_path = os.path.join(mr_volume_dir, files)
    mr_image_meta = sitk.ReadImage(mr_images_path)
    
    image_size = mr_image_meta.GetSize()
    size_list += [image_size] #Image size information of each MR image in the dataset
                                #(X, Y, Z), i.e. (width, height, # of slices) for T1 and T2 images
                                #(Z, Y, X), i.e. (# of slices, height, width) for images with SPACE sequence
    

# Investigating segmentation masks
mr_masks_dir = ("pathtodataset/masks") #Remember to adjust it according to your local path
mr_masks = os.listdir(mr_masks_dir)
mr_maskfiles = []

## To get rid of the redundant data
for mask in mr_masks:
    if mask[0].isnumeric():
        mr_maskfiles += [mask]

sorted_mr_maskfiles = sorted(mr_maskfiles, key=extract_number)

mr_mask_labels = []
size_list2 = []
for masks in sorted_mr_maskfiles:
    mr_masks_path = os.path.join(mr_masks_dir, masks)
    mr_mask_meta = sitk.ReadImage(mr_masks_path)
    mr_mask_metainf = sitk.GetArrayFromImage(mr_mask_meta)
    
    mr_mask_label = np.unique(mr_mask_metainf) # Unique values in this array corresponds to segmentation labels
    mr_mask_labels.append(mr_mask_label)
    
    # An alternative way to extract information about number of slices and pixel size
    image_size2 = mr_mask_metainf.shape
    size_list2.append(image_size2) # (Z, Y, X), i.e. (# of slices, height, width)
    
    
    
# Cropping out the non-spine voxels
mr_image_path1 = '/Volumes/WD Elements/Pattern Recognition/Project/images/1_t1.mha'
mr_mask_path1 = '/Volumes/WD Elements/Pattern Recognition/Project/masks/1_t1.mha'

mr_image1 = sitk.ReadImage(mr_image_path1)
mask_image1 = sitk.ReadImage(mr_mask_path1)

mr_arr1 = sitk.GetArrayFromImage(mr_image1)
mask_arr1 = sitk.GetArrayFromImage(mask_image1)

    
#Method 1
non_zero_coords = np.argwhere(mask_arr1)
start = non_zero_coords.min(axis=0)
end = non_zero_coords.max(axis=0) + 1  # +1 to include the end slice

cropped_mr_image = sitk.GetArrayFromImage(mr_image1)[start[0]:end[0], start[1]:end[1], :]

# Converting the cropped image to a SimpleITK image
cropped_mr_image_sitk = sitk.GetImageFromArray(cropped_mr_image)

# Saving the cropped MR image
output_path = '/Volumes/WD Elements/Pattern Recognition/Project/output/cropped_mr_image2.mha'  # Replace with the desired output path
sitk.WriteImage(cropped_mr_image_sitk, output_path)


#Method 2
cropped_imarr = np.zeros_like(mr_arr1)
for s in range(mask_arr1.shape[2]):
    for r in range(mask_arr1.shape[0]):
        for c in range(mask_arr1.shape[1]):
            if mask_arr1[r, c, s] > 0:
                cropped_imarr[r, c, s] = mr_arr1[r, c, s]

# Converting the cropped array back to a SimpleITK image
cropped_imarr_sitk = sitk.GetImageFromArray(cropped_imarr)

# Setting the origin, spacing, and direction of the cropped image to match the original image
cropped_imarr_sitk.SetOrigin(mr_image1.GetOrigin())
cropped_imarr_sitk.SetSpacing(mr_image1.GetSpacing())
cropped_imarr_sitk.SetDirection(mr_image1.GetDirection())

cropped_imarr_sitk = sitk.GetImageFromArray(cropped_imarr)
output_path = '/Volumes/WD Elements/Pattern Recognition/Project/output/cropped_mr_image3.mha'  # Replace with the desired output path
sitk.WriteImage(cropped_imarr_sitk, output_path)
