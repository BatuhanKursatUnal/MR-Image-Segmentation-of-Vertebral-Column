# Pre-processing script
import os
import configparser
from vertebraeseg import extract_number_and_modality, extract_image_size, extract_mask_labels, pre_process_mr_images

config = configparser.ConfigParser()
config.read('configuration.txt') #It must be stored in the same folder as the script

# Importing the paths from configuration.txt file
mr_volume_dir = config['paths']['mr_volume_dir']
mr_masks_dir = config['paths']['mr_masks_dir']
size_list_path  = config['paths']['size_list_path']
cropped_path = config['paths']['cropped_path']
mr_inputfolder_path = config['paths']['mr_inputfolder_path']
mr_outputfolder_path  = config['paths']['mr_outputfolder_path']
masks_outputfolder_path = config['paths']['masks_outputfolder_path']
desired_order = config['settings']['desired_order']

# To get rid of the redundant data in images
mr_images = os.listdir(mr_volume_dir)
size_list = []
mr_imagefiles = []
for file in mr_images:
    if file[0].isnumeric():
        mr_imagefiles.append(file)
        
# Sorting out the image names
sorted_mr_imagefiles = sorted(mr_imagefiles, key=extract_number_and_modality)
extract_image_size(sorted_mr_imagefiles) #Extracting image sizes

# Investigating segmentation masks
mr_masks = os.listdir(mr_masks_dir)
mr_maskfiles = []

# To get rid of the redundant data in masks
for mask in mr_masks:
    if mask[0].isnumeric():
        mr_maskfiles.append(mask)

# Sorting out the mask names
sorted_mr_maskfiles = sorted(mr_maskfiles, key=extract_number_and_modality)
extract_mask_labels(sorted_mr_maskfiles) #Extracting mask sizes

# Running the preprocessing pipeline
pre_process_mr_images(sorted_mr_imagefiles, mr_volume_dir, cropped_path, size_list_path, 
                         mr_inputfolder_path, mr_outputfolder_path, mr_masks_dir, 
                         masks_outputfolder_path, desired_order)
