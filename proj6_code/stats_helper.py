import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from image_loader import ImageLoader

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
    '''
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    '''

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################                
    gray_images = []
    
    # Loop through all files in the directory
    for root, dirs, files in os.walk(dir_name): # incase there are sub-directories
        for file in files:
            if ".jpg" in file or ".JPG" in file:
                # Load image in grayscale and scale to [0,1]
                image = Image.open(os.path.join(root, file)).convert('L')
                image = np.array(image, dtype=np.float32) / 255.0

                # Add image data to the list
                gray_images.extend(image.flatten())

    # Convert the list to a numpy array
    gray_images = np.array(gray_images).reshape(-1, 1)
#     print("image shapes combined", gray_images.shape)

    # Compute mean and standard deviation using StandardScaler
    scaler = StandardScaler()
    scaler.fit(gray_images)

    mean = scaler.mean_
    std = scaler.scale_
    
    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
