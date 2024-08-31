![Static Badge](https://img.shields.io/badge/code-Python-red?style=flat-square)
[![Dependencies](https://img.shields.io/badge/dependencies-SimpleITK-blue)](https://pypi.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-PyTorch-darkblue?style=plastic)](https://pypi.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-pandas-lightblue?style=plastic)](https://pypi.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-scikitlearn-white?style=plastic)](https://pypi.org/)


# MR Image Segmentation of Vertebral Column

Vertebral column of humans contain three main structures: vertebrae, intervertebral discs, and the spinal canal. Segmenting the vertebral column images acquired from medical imaging techniques (MR scans in this case) provides the clinicians with the ability to correctly identify any abnormalities, like fractures, tumors, or diseases like scoliosis and plan the possible treatments, surgeries and the targeted regions in these operations.

Traditionally, the segmentation of the vertebral column is performed in a time consuming and intensive fashion by employing people to manually do the task. Automatic segmentation algorithms perform this task much more efficiently and quickly. They are also quite beneficial with the handling of large amounts of data in this area. Finally, when implemented well, they give accurate results with small amount of mistakes. 

In this project, I used the Spider Dataset, which is publicly available, and built a U-Net architecture using PyTorch and other necessary dependencies to automatically segment the 3D MR images that are pre-processed again by me mainly by exploiting the SimpleITK functions.

## Data

The dataset used for this study contains 3D multi-center lumbar spine MR images collected from patients with lower back pain with the aim of segmenting spinal canal, vertebrae, and intervertebral discs. 

Data collected from 257 patients contained at most 3 MRI series. Out of these 257, 218 studies including 447 MRI series are made publicly available. 447 T1 and T2-weighted MR images are collected from the 218 patients with pain in their lower back, of which 63% were female.  After the necessary data preparation and cleaning, 358 of these images were found appropriate to use in this project.

The Spider dataset contains both the images and corresponding masks for each image, which constitute multi-labeled classes for segmentation.

The dataset could be found on the following website: “https://spider.grand-challenge.org/data/“


## Project Outline

This is an image segmentation project that runs on Python and includes two main parts: preprocessing of the data and the U-Net model and implementation.

1. Data preparation and cleaning are performed in the pre-processing step. This part is contained in the [vertebraeseg.py](./vertebraeseg.py) file and it includes the following pre-processing steps:
     - Data cleaning and storage regulation.
     - Intensity normalization; min-max normalization or z-score normalization, input dependent.
     - Cropping out the non-spinal parts of the images and masks, using thresholding, morphological operations and connected component analysis.
     - Resizing all the images and masks into a single, consistent size, which is chosen as (256, 128, 32) [height, width, slice].

2. The second step is allocated to building up the neural network architecture contains the files containing the architecture of the network, training and validation sets and finally the test set.
     - The [unet_model.py](./unet_model.py) file contains the UNet architecture with the determined feature dimensions.
     - The [train.py](./train.py) consist of two main parts: training and validation. They are executed under the same iterative loop over the epochs and numerous metrics are calculated, such as loss, accuracy, and F1 score. Cross entropy loss function is used and as optimizer, I used AdamW, which after couple of trial and errors showed that it performs better and faster than the Adam optimizer for this project.
     - Lastly, [test_unet.py](./test_unet.py) consists of the testing loop of the model.

3. Finally, in [tests.py](./tests.py) file, one could find the testing performed on the functions written in the other files.


## User Manual






## Results

This section contains some of the resulting masks together with the corresponding image and ground truth mask. In addition, it contains the loss and accuracy plots for the training and validation sets.



### Bibliography

“Spider - Grand Challenge.” Grand Challenge, spider.grand-challenge.org/data/. Accessed 31 Aug. 2024. 