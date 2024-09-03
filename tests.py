# Testing
import pytest
import numpy as np
import SimpleITK as sitk
from vertebraeseg import CropNonspinalVoxels, IntensityNormalizer, resize_image
import configparser
import os

config = configparser.ConfigParser()

# Creating dummy image path for testing
@pytest.fixture(scope="module")
def test_image_path():
    
    '''
    Dummy image path creator.
    
    Yields
    ------
    testimage_path: str
        Path to the dummy image.

    '''
    testimage_path = config['paths']['testimage_path']
    test_image = sitk.Image([10, 10, 10], sitk.sitkFloat32)
    test_image_array = np.random.rand(10, 10, 10)
    test_image = sitk.GetImageFromArray(test_image_array)
    sitk.WriteImage(test_image, testimage_path)
    yield testimage_path
    os.remove(testimage_path)
    

# Creating dummy mask path for testing
@pytest.fixture(scope="module")
def test_mask_path():
    
    '''
    Dummy mask path creator.

    Yields
    ------
    testimage_path: str
        Path to the dummy mask.

    '''
    testmask_path = config['paths']['testmask_path']
    test_mask_array = np.random.randint(0, 2, size=(10, 10, 10)).astype(np.uint8)
    test_mask = sitk.GetImageFromArray(test_mask_array)
    sitk.WriteImage(test_mask, testmask_path)
    yield testmask_path
    os.remove(testmask_path)



class TestInputValidation:
    
    '''
    A class for testing input validation of CropNonspinalVoxels function

    Methods:
    --------
    testwith_both_valid(test_image_path):
        Tests the inputs with both valid image and valid array.

    testwith_invalid_image(test_image_path):
        Tests the inputs with invalid image and valid array.
        
    testwith_invalid_array(test_image_path):
        Tests the inputs with valid image but invalid array.
    
    testwith_empty_image():
        Tests the inputs with an empty dummy image.
        
    '''
    def testwith_both_valid(self, test_image_path):
        
        '''
        Tests the CropNonspinalVoxels class with both valid image and array inputs.
    
        This test verifies that the CropNonspinalVoxels class correctly processes
        valid inputs. A SimpleITK image and a corresponding NumPy array are passed
        to the class, and it is expected to return a cropped SimpleITK image without
        raising any exceptions.
    
        Parameters
        ----------
        test_image_path: str
            The file path to a valid test image.
    
        Returns
        -------
        None.
    
        Assertions
        ----------
        Asserts that the result of the cropping operation is an instance of SimpleITK.Image.
        '''
        dummy_image = sitk.ReadImage(test_image_path)
        dummy_arr = sitk.GetArrayFromImage(dummy_image)
        
        cropper = CropNonspinalVoxels(dummy_image, dummy_arr)
        cropped_image_result = cropper.crop_nonspinal()
        assert isinstance(cropped_image_result, sitk.Image)
    
    def testwith_invalid_image(self, test_image_path):
        
        '''
        Tests the CropNonspinalVoxels class with an invalid image input.
    
        This test checks that the CropNonspinalVoxels class raises a TypeError
        when the first input (the image) is invalid. A dummy array is used in place
        of a valid SimpleITK image to simulate an invalid image input scenario. 
    
        Parameters
        ----------
        test_image_path: str
            The file path to a valid test image. 
            
        Returns
        -------
        None.
    
        Raises
        ------
        TypeError
            If the first input to CropNonspinalVoxels is not a valid SimpleITK image.
    
        '''
        dummy_image = sitk.ReadImage(test_image_path)
        dummy_arr = sitk.GetArrayFromImage(dummy_image)
        dummy_valid_array = np.zeros((10, 10, 10), dtype= np.uint8)
        
        with pytest.raises(TypeError):
            CropNonspinalVoxels(dummy_valid_array, dummy_arr)
        
    def testwith_invalid_array(self, test_image_path):
        
        '''
        Tests the CropNonspinalVoxels class with an invalid array input.
    
        This test checks that the CropNonspinalVoxels class raises a TypeError
        when the second input (the array) is invalid. A string is used in place
        of a valid NumPy array to simulate an invalid array input scenario.
    
        Parameters
        ----------
        test_image_path: str
            The file path to a valid test image. This image is used to create a 
            valid SimpleITK image for testing.
    
        Returns
        -------
        None.
    
        Raises
        ------
        TypeError
            If the second input to CropNonspinalVoxels is not a valid NumPy array.
            '''
        dummy_image = sitk.ReadImage(test_image_path)
        dummy_invalid_array = "not_an_array"
        
        with pytest.raises(TypeError):
            CropNonspinalVoxels(dummy_image, dummy_invalid_array)
            
    def test_empty_image(self):
        
        '''
        Tests the CropNonspinalVoxels class with an empty image input.
    
        Parameters
        ----------
        None.
    
        Returns
        -------
        None.
    
        Raises
        ------
        ValueError
            If the input image has no non-zero components, indicating an empty image.
            
        '''
        dummy_valid_array = np.zeros((10, 10, 10), dtype= np.uint8)
        dummy_zero_image = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        
        with pytest.raises(ValueError):
            cropper = CropNonspinalVoxels(dummy_zero_image, dummy_valid_array)
            cropper.crop_nonspinal()
            
     
# Testing the intensity normalizer object
def test_minmax_normalizer(test_image_path):
    
    '''
    Testing the minmax normalizer method of IntensityNormalizer object.
    
    It checks for output type, and range of the output intensity values to see
    if the normalizer performs the intended task.

    Parameters
    ----------
    test_image_path: str
        Path to the test image.

    Returns
    -------
    None.
    
    Assertions
    ----------
    Asserts that the result of the z-score normalization operation is an instance
    of SimpleITK.Image and NumPy array, respectively.
    Asserts the output intensity values are between 0-255.

    '''
    normalizer = IntensityNormalizer(test_image_path)
    mr_image_normalized, mr_array_normalized = normalizer.minmax_normalizer()
    
    assert isinstance(mr_image_normalized, sitk.Image)
    assert isinstance(mr_array_normalized, np.ndarray)
    assert np.all(mr_array_normalized >= 0)
    assert np.all(mr_array_normalized <= 255)

def test_zscore_normalizer(test_image_path):
    
    '''
    Testing the z-score normalizer method of IntensityNormalizer object.
    
    It checks for output type, and range of the output intensity values to see
    if the normalizer performs the intended task.

    Parameters
    ----------
    test_image_path : str
        Path to the test image.

    Returns
    -------
    None.
    
    Assertions
    ----------
    Asserts that the result of the z-score normalization operation is an instance
    of SimpleITK.Image and NumPy array, respectively.
    Asserts the mean and standard deviation values close to 0 and 1, respectively.

    '''
    normalizer = IntensityNormalizer(test_image_path)
    mr_image_znormalized, mr_array_znormalized = normalizer.zscore_normalizer()
    
    assert isinstance(mr_image_znormalized, sitk.Image)
    assert isinstance(mr_array_znormalized, np.ndarray)
    assert np.isclose(np.mean(mr_array_znormalized), 0, atol=1e-1)
    assert np.isclose(np.std(mr_array_znormalized), 1, atol=1e-1)

def test_invalid_path():
    '''
    Checks if the input path, i.e. path to the test image, has the correct type,
    i.e. string.

    Returns
    -------
    None.
    
    Raises
    -------
    TypeError 
        When input path is not valid.

    '''
    with pytest.raises(TypeError):
        IntensityNormalizer(123)
            
         
          
# Testing the image resizing function
def test_resize_image(test_image_path):
    '''
    Tests if the image resizing works as intended.

    Parameters
    ----------
    test_image_path : str
        Path to the test image.

    Returns
    -------
    None.
    
    Assertions
    ----------
    Asserts that the output path exists, the resulting size of the image is the
    same with the desired size, and resized image has the same pixel type as
    the input.
    

    '''
    testresize_out_path = config['paths']['testresize_out_path']
    desired_size = (5, 5, 5)

    resize_image(test_image_path, output_path, desired_size)

    assert os.path.exists(testresize_out_path)

    # Load the resized image
    resized_image = sitk.ReadImage(testresize_out_path)
    resized_size = resized_image.GetSize()

    assert resized_size == desired_size

    # Check if the resized image has the same pixel type as the original
    original_image = sitk.ReadImage(test_image_path)
    assert resized_image.GetPixelID() == original_image.GetPixelID()

    os.remove(testresize_out_path)

def test_resize_image_invalid_input():
    '''
    Tests the image resizing function with invalid input.

    Returns
    -------
    None.
    
    Raises
    -------
    RuntimeError 
        When trying to resize an image from a non-existent file path.

    '''
    with pytest.raises(RuntimeError):
        resize_image("non_existent_image.mha", testresize_out_path, (5, 5, 5))
            

if __name__ == "__main__":
    pytest.main()