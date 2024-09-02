# Pre-processing functions and classes
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import re
        
def extract_number_and_modality(mri_file):
    
    '''
    Key function to determine sorting dependencies

    Parameters
    ----------
    mri_file : list of str
        List of image files of the dataset

    Returns
    -------
    subject_number : int
        Subject number, i.e. first integer in data name.
        
    modality_priority: dict
        Priority determined by the way data is stored in the folder.

    '''
    match = re.match(r"(\d+)_(t1|t2_SPACE|t2)\.mha", mri_file)
    subject_number = int(match.group(1))
    modality = match.group(2)
    # Assigning different priority to each modality to enforce the same order as in the folder
    modality_priority = {"t1": 1, "t2_SPACE": 2, "t2": 3}
    return subject_number, modality_priority[modality]


def extract_image_size(image_list):
    
    '''
    Extracts the sizes of images in a specified path, lists them, and creates a
    dataframe containing all the sizes.

    Parameters
    ----------
    image_list: list of str
        List of image names represented as strings.

    Returns
    -------
    size_list_df: pandas.DataFrame
        List of dimensions (sizes) of images stored in a dataframe.

    '''
    for files in image_list:    
        mr_images_path = os.path.join(mr_volume_dir, files)
        mr_image_meta = sitk.ReadImage(mr_images_path)
        
        image_size = mr_image_meta.GetSize()
        size_list += [image_size] #Image size information of each MR image in the dataset
        size_list_df = pd.DataFrame(size_list)
        size_list_df.to_csv(size_list_path)
        
    return size_list_df


def extract_mask_labels(mask_list):
    
    '''
    Extracts the unique values in the metadata of each mask which correspond to
    the labels contained in them.

    Parameters
    ----------
    mask_list: list of str
        List of mask names represented as strings.

    Returns
    -------
    mr_mask_labels: list
        List containing all the labels for each mask.

    '''
    mr_mask_labels = []
    size_list2 = []
    for masks in sorted_mr_maskfiles:
        mr_masks_path = os.path.join(mr_masks_dir, masks)
        mr_mask_meta = sitk.ReadImage(mr_masks_path)
        mr_mask_metainf = sitk.GetArrayFromImage(mr_mask_meta)
        
        mr_mask_label = np.unique(mr_mask_metainf) # Unique values in this array corresponds to segmentation labels
        mr_mask_labels.append(mr_mask_label)
        
    return mr_mask_labels

    
# Pixel intensity normalization using min-max and z-score normalization methods
class IntensityNormalizer:
    
    '''
    A class for pixel intensity normalization on MR images using min-max and/or
    z-score normalization methods.

    Attributes:
    -----------
    mr_image_path: str
        The file path to the MR image that needs to be normalized.

    Methods:
    --------
    minmax_normalizer(): 
        Normalizes the image using min-max normalization.

    zscore_normalizer(): 
        Normalizes the image using z-score normalization.
        
    '''
    
    def __init__(self, mr_image_path):
        
        '''
        Constructs the IntensityNormalizer with the specified image path.

        Parameters
        ----------
        mr_image_path: str
            The file path to the MR image that needs to be normalized.

        Raises:
        -------
        TypeError: 
            If the input `mr_image_path` is not a string.
            
        Returns
        -------
        None.
        
        '''
        if not isinstance(mr_image_path, str):
            raise TypeError("Input is not a string")
        self.mr_image_path = mr_image_path
    
    def minmax_normalizer(self):
        
        '''
        Applies min-max normalization to the inputted MR image.
        
        Min-max normalization rescales the pixel intensity values to the 
        range [0, 255] and converts the image to 8-bit.

        Returns
        -------
        mr_image_normalized : SimpleITK.Image
            The min-max normalized image.
        mr_array_normalized : numpy.ndarray
            The array of min-max normalized image
            
        '''
        
        mr_image = sitk.ReadImage(self.mr_image_path)
        mr_array = sitk.GetArrayFromImage(mr_image)
        
        mr_image_normalized = sitk.Cast(sitk.RescaleIntensity(mr_image), sitk.sitkUInt8)
        mr_array_normalized = sitk.GetArrayFromImage(mr_image_normalized)
        
        return mr_image_normalized, mr_array_normalized
    
    def zscore_normalizer(self):
        
        '''
        Applies z-score normalization to the inputted MR image.
        
        Z-score normalization transforms the pixel intensity values so that they
        have the mean 0, and standard deviation 1.

        Returns
        -------
        mr_image_znormalized : SimpleITK.Image
            The image normalized with z-score method
        mr_array_znormalized : numpy.ndarray
            The array of image normalized with z-score method
            
        '''
        
        mr_image = sitk.ReadImage(self.mr_image_path)
        mr_array = sitk.GetArrayFromImage(mr_image)
        
        mr_mean_int = np.mean(mr_array)
        mr_std_int = np.std(mr_array)
        
        mr_array_znormalized = (mr_array - mr_mean_int)/mr_std_int
        mr_image_znormalized = sitk.GetImageFromArray(mr_array_znormalized)
        
        return mr_image_znormalized, mr_array_znormalized
    
    
# Cropping out the non-spine voxels
class CropNonspinalVoxels:
    
    '''
    A class for cropping out non-spinal voxels from the input MR image.

    Attributes:
    -----------
    mr_image_inp: SimpleITK.Image
        The image input
        
    mr_array_inp: numpy.ndarray
        The array of image input
        
    Methods:
    --------
    crop_nonspinal():
        Crops out non-spinal voxels of the image
        
    '''
    
    def __init__(self, mr_image_inp, mr_array_inp):
        
        '''
        Constructs the CropNonspinalVoxels with specified paths to image and array.

        Parameters
        ----------
        mr_image_inp: SimpleITK.Image
            The image input
            
        mr_array_inp: numpy.ndarray
            The array of image input

        Raises
        ------
        TypeError:
            If the 'mr_image_inp' is not of form SimpleITK.Image
            If the 'mr_array_inp' is not of form numpy.ndarray

        Returns
        -------
        None.

        '''
        if not isinstance(mr_image_inp, sitk.Image):
            raise TypeError("Input mr_image_inp should be a SimpleITK Image")
        if not isinstance(mr_array_inp, np.ndarray):
            raise TypeError("Input mr_array_inp should be a numpy array")
            
        #Casting
        self.mr_image = mr_image_inp
        self.mr_array = mr_array_inp
    
    def crop_nonspinal(self):
        
        '''
        Crops out non-spinal voxels using thresholding, morphological operations and connected component analysis    

        Parameters
        ----------
        mr_image_inp: SimpleITK.Image
            The image input
            
        mr_array_inp: numpy.ndarray
            The array of image input
            
        Raises
        ------
        ValueError:
            When the size of the array representing connected components is 0,
            i.e. there is no connected components in the image

        Returns
        -------
        cropped_mr_im: SimpleITK.Image
            Resulting MR image with non-spinal voxels cropped out
        
        '''
        
        #Thresholding
        #Otsu thresholding to indicate the optimum lower threshold value
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        thresholding = otsu_filter.Execute(self.mr_image)
        thresholding_val = otsu_filter.GetThreshold()
        th_im = sitk.BinaryThreshold(self.mr_image, lowerThreshold=thresholding_val, upperThreshold=255, insideValue=1, outsideValue=0) #Thresholded image

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

# Resizing function
def resize_image(image_path, output_path, desired_size):
    
    '''
    Resizes images to a specified size to achieve consistency

    Parameters
    ----------
    image_path : str
        Input image path
        
    output_path : str
        Output image path for resized image
        
    desired_size: tuple
        Specify the size in 3D

    Returns
    -------
    None.

    '''
    mr_image = sitk.ReadImage(image_path)
    input_size = mr_image.GetSize()
    input_spacing = mr_image.GetSpacing()
    input_direction = mr_image.GetDirection()
    input_origin = mr_image.GetOrigin()
    
    scaling_factor = [float(input_size[i]) / desired_size[i] for i in range(3)]
    output_spacing = [input_spacing[i] * scaling_factor[i] for i in range(3)]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(desired_size)
    resample.SetOutputDirection(input_direction)
    resample.SetOutputOrigin(input_origin)
    resample.SetInterpolator(sitk.sitkLinear)
    
    resized_image = resample.Execute(mr_image)
    sitk.WriteImage(resized_image, output_path)

    
# Preprocessing Pipeline
def preprocess_mr_images(image_list, mr_volume_dir, cropped_path, size_list_path, 
                         mr_inputfolder_path, mr_outputfolder_path, mr_masks_dir, 
                         masks_outputfolder_path, desired_order):
    
    """
    Processes MR images and masks by normalizing, cropping, and resizing them.

    Parameters
    ----------
    image_list: list of str
        List of sorted MR image filenames.
        
    mr_volume_dir: str
        Directory containing MR image volumes.
        
    cropped_path: str
        Output path to save cropped images.
        
    size_list_path: str
        Path to the CSV file with size information.
        
    mr_inputfolder_path: str
        Input folder path for resizing images.
        
    mr_outputfolder_path: str
        Output folder path for resized images.
        
    mr_masks_dir: str
        Directory containing MR masks.
        
    masks_outputfolder_path: str
        Output folder path for resized masks.
        
    desired_order: tuple
        Desired dimensions of the image that will be specified in resizing function.
    """
    norm_type = input('Determine the type of normalizaion (minmax or zscore):')
    for files in image_list:
        mr_images_path = os.path.join(mr_volume_dir, files)
        #Z-score Normalization
        if norm_type == 'zscore':
            mr_image_path1 = mr_images_path
            normalizer = IntensityNormalizer(mr_image_path1)
            znormalized_image, znormalized_array = normalizer.zscore_normalizer() #Z-score normalization
            mr_image = znormalized_image
            mr_array = znormalized_array
            
        #Min-max normalization
        if norm_type == 'minmax':
            mr_image_path1 = mr_images_path
            normalizer = IntensityNormalizer(mr_image_path1)
            normalized_image, normalized_array = normalizer.minmax_normalizer() #Min-max normalization
            mr_image = normalized_image
            mr_array = normalized_array    
        
        #Cropping out
        cropper = CropNonspinalVoxels(mr_image, mr_array)
        cropped_mr_image = cropper.crop_nonspinal()
        output_path = cropped_path + 'cropped_' + files
        sitk.WriteImage(cropped_mr_image, output_path)

    # Eliminate images with inconsistent axis sizes
    size_list_cr_df = pd.read_csv(size_list_path)
    size_list_cr_df_new = size_list_cr_df[(size_list_cr_df.iloc[:, 1] <= 40) | (size_list_cr_df.iloc[:, 1] == 120)]
    size_list_cr_df_new.reset_index(drop=True, inplace=True)

    sorted_mr_imagefiles_new = []
    for index in size_list_cr_df_new.iloc[:, 0]:
        sorted_mr_imagefiles_new.append(image_list[index])
        
    # Resize all images to a standard size
    for image in sorted_mr_imagefiles_new:
        resizing_input_path = mr_inputfolder_path + 'cropped_' + image
        resizing_output_path = mr_outputfolder_path + 'rs_' + image
        resize_image(resizing_input_path, resizing_output_path, desired_order)

    # Resizing all the masks to a standard size
    for image in sorted_mr_imagefiles_new:
        resizingmasks_input_path = mr_masks_dir + image
        resizingmasks_output_path = masks_outputfolder_path + 'rs_mask_' + image
        resize_image(resizingmasks_input_path, resizingmasks_output_path, desired_order)
    
    
    
