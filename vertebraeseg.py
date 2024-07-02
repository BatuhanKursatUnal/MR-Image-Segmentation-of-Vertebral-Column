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