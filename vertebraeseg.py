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
        mr_imagefiles.append(file)
        
def extract_number_and_modality(mri_file):
    '''
    Key function to determine sorting dependencies

    Parameters
    ----------
    mri_file : Files of the dataset stored in your local machine

    Returns
    -------
    subject_number : Subject number, i.e. first integer in data name
    modality_priority: Priority determined by the way data is stored in the folder

    '''
    match = re.match(r"(\d+)_(t1|t2_SPACE|t2)\.mha", mri_file)
    subject_number = int(match.group(1))
    modality = match.group(2)
    # Assigning different priority to each modality to enforce the same order as in the folder
    modality_priority = {"t1": 1, "t2_SPACE": 2, "t2": 3}
    return subject_number, modality_priority[modality]

sorted_mr_imagefiles = sorted(mr_imagefiles, key=extract_number_and_modality)
    
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
        mr_maskfiles.append(mask)

sorted_mr_maskfiles = sorted(mr_maskfiles, key=extract_number_and_modality)

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
    
    
# Pixel intensity normalization using min-max and z-score normalization methods
class IntensityNormalizer:
    '''
    Intensity normalizer takes in image path, determined by the user, and applies min-max and z-score normalization methods.
    '''
    def __init__(self, mr_image_path):
        if not isinstance(mr_image_path, str):
            raise TypeError("Input is not a string")
        self.mr_image_path = mr_image_path
    
    def minmax_normalizer(self):
        '''
        
        Parameters
        ----------
        mr_image_path: Path of the MR image in your local machine
        
        Returns
        -------
        mr_image_normalized: Normalized image
        mr_array_normalized: Normalized array

        '''
        
        mr_image = sitk.ReadImage(self.mr_image_path)
        mr_array = sitk.GetArrayFromImage(mr_image)
        
        mr_image_normalized = sitk.Cast(sitk.RescaleIntensity(mr_image), sitk.sitkUInt8)
        mr_array_normalized = sitk.GetArrayFromImage(mr_image_normalized)
        
        return mr_image_normalized, mr_array_normalized
    
    def zscore_normalizer(self):
        '''
        Parameters
        ----------
        mr_image_path: Path of the MR image in your local machine

        Returns
        -------
        mr_image_znormalized: Normalized image using z-score normalization
        mr_array_znormalized: Normalized array using z-score normalization
        '''
        
        mr_image = sitk.ReadImage(self.mr_image_path)
        mr_array = sitk.GetArrayFromImage(mr_image)
        
        mr_mean_int = np.mean(mr_array)
        mr_std_int = np.std(mr_array)
        
        mr_array_znormalized = (mr_array - mr_mean_int)/mr_std_int
        mr_image_znormalized = sitk.GetImageFromArray(mr_array_znormalized)
        
        return mr_image_znormalized, mr_array_znormalized
    
    
# Cropping out the non-spine voxels
#Method 3
class CropNonspinalVoxels:
    
    def __init__(self, mr_image_inp, mr_array_inp):
        if not isinstance(mr_image_inp, sitk.Image):
            raise TypeError("Input mr_image_inp should be a SimpleITK Image")
        if not isinstance(mr_array_inp, np.ndarray):
            raise TypeError("Input mr_array_inp should be a numpy array")
            
        #Casting
        self.mr_image = sitk.Cast(sitk.RescaleIntensity(mr_image_inp), sitk.sitkUInt8)
        self.mr_array = mr_array_inp
    
    def crop_nonspinal(self):
        
        '''
        Crops out non-spinal voxels using thresholding, morphological operations and connected component analysis    

        Parameters
        ----------
        mr_image_inp : Raw MR image
        mr_array_inp : Array extracted from raw MR image

        Returns
        -------
        MR image with non-spinal voxels cropped out
        '''
        
        #Thresholding
        th_im = sitk.BinaryThreshold(self.mr_image, lowerThreshold=175, upperThreshold=255, insideValue=1, outsideValue=0) #Thresholded image

        #Morphological Operation
        radius_morph = [1, 1, 1]
        ##Closing (Gives the best results)
        th_im_closed = sitk.BinaryMorphologicalClosing(th_im, radius_morph) #Image that went through closing operation
    
        #Connected Component Analysis
        ##Labeling connected components
        con_comps = sitk.ConnectedComponent(th_im_closed) #Connected components
        cc_array = sitk.GetArrayFromImage(con_comps)
        
        if cc_array.max() == 0:
            raise ValueError("No connected components found")

        ##The largest connected component
        unique, counts = np.unique(cc_array, return_counts=True)
        largest_comp = unique[np.argmax(counts[1:]) + 1]

        ##Creating a mask for the largest component
        largest_comp_mask = (cc_array == largest_comp).astype(np.uint8)
        largest_comp_im = sitk.GetImageFromArray(largest_comp_mask)

        ##Creating the bounding box around the largest component
        non_zero_coords = np.argwhere(largest_comp_mask > 0)
    
        start = non_zero_coords.min(axis=0)
        end = non_zero_coords.max(axis=0) + 1  # +1 to include the end slice

        ##Cropping the original MR image using the bounding box coordinates
        cropped_mr_arr = self.mr_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        cropped_mr_im = sitk.GetImageFromArray(cropped_mr_arr)

        ##Matching the original image
        cropped_mr_im.SetOrigin(self.mr_image.GetOrigin())
        cropped_mr_im.SetSpacing(self.mr_image.GetSpacing())
        cropped_mr_im.SetDirection(self.mr_image.GetDirection())
    
        return cropped_mr_im
    