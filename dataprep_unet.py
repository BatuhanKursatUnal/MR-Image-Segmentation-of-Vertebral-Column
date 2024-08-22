# Data Preparation for U-Net

import os
import re
import pandas as pd

mr_volume_dir = ('/Path/to/images') #Path to the images
mr_images = os.listdir(mr_volume_dir) 
size_list = []
mr_imagefiles = []

# To get rid of the redundant data
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

# Sort based on the numeric part and then the modality
sorted_mr_imagefiles = sorted(mr_imagefiles, key=extract_number_and_modality)

# Eliminate images with inconsistent axis sizes
size_list_cr_df = pd.read_csv('/Path/to/sizelist.csv') #Path to the list of sizes of images for indexing
size_list_cr_df_new = size_list_cr_df[(size_list_cr_df.iloc[:, 1] <= 40) | (size_list_cr_df.iloc[:, 1] == 120)]
size_list_cr_df_new.reset_index(drop=True, inplace=True)

#Sorting out the file names properly
sorted_mr_imagefiles_new = []
for index in size_list_cr_df_new.iloc[:, 0]:
    sorted_mr_imagefiles_new.append(sorted_mr_imagefiles[index])
    
images_volume_dir = '/Path/to/resized_v5' #Path to pre-processed images folder
masks_dir = '/Path/to/resized_masks_v1' #Path to pre-processed masks folder
