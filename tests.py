# Testing

import pytest
import unittest
from unittest import TestCase
import numpy as np
import SimpleITK as sitk

from vertebraeseg import CropNonspinalVoxels
dummy_zero_image = sitk.Image([10, 10, 10], sitk.sitkUInt8)
dummy_valid_array = np.zeros((10, 10, 10), dtype= np.uint8)
mr_image_path1 = mr_pathto_image
mr_image1 = sitk.ReadImage(mr_image_path1)
mr_array1 = sitk.GetArrayFromImage(mr_image1)
class TestInputValidation:
    '''
    This testing scheme checks the validity of inputs on CropNonSpinalVoxels class
    '''
    def testwith_both_valid(self):
        '''
        Tests when both image and array are valid
        '''
        cropper = CropNonspinalVoxels(mr_image1, mr_array1)
        cropped_image_result = cropper.crop_nonspinal()
        assert isinstance(cropped_image_result, sitk.Image)
    
    def testwith_invalid_image(self):
        '''
        Tests when image is invalid but array is valid
        '''
        with pytest.raises(TypeError):
            CropNonspinalVoxels(dummy_valid_array, mr_array1) #For the first input, dummy_valid_array becomes an invalid image.
        
    def testwith_invalid_array(self):
        '''
        Tests when image is valid but array is invalid
        '''
        with pytest.raises(TypeError):
            CropNonspinalVoxels(mr_image1, dummy_zero_image) #For the second input, dummy_valid_image becomes an invalid array.
            
    def test_empty_image(self):
        '''
        Tests when there are no non-zero components in the input image, hence empty image
        '''
        with pytest.raises(ValueError):
            cropper = CropNonspinalVoxels(dummy_zero_image, dummy_valid_array)
            cropper.crop_nonspinal()
            
if __name__ == "__main__":
    pytest.main()