########################################################################################################

import rawpy
import numpy as np
from .constants import RED_POSITION , BLUE_POSITION , GREEN_POSITION
import matplotlib.pyplot as plt
########################################################################################################

"""

1. Whenever you are reading a DNG file , use raw_image_visible to get the image data.
2. if you use raw.postprocess() to get the image data, it will convert the raw image to RGB image.

"""

########################################################################################################

def read_dng_image(file_path):
    """
    Read a DNG format image file.
    
    Args:
        file_path (str): Path to the DNG file
        
    Returns:
        numpy.ndarray: Image data as RGB array
    """
    with rawpy.imread(file_path) as raw:
        raw_image = raw.raw_image_visible
    return raw_image

########################################################################################################

def get_bayer_pattern(file_path):
    """

        Get the Bayer Pattern type from a DNG file.

        Args:
            file_path (str): Path to the DNG file

        Returns:
            tuple: Bayer pattern as (R, G1, G2, B) values
            str: Description of the pattern
    """

    with rawpy.imread(file_path) as raw:
        pattern = raw.raw_pattern
        pattern_desc = raw.color_desc
    
    return pattern , pattern_desc

########################################################################################################
def get_cfa_mask_rgbg(height , width):
    cfa_mask = np.zeros((height , width))
    
    cfa_mask[0 : height: 2 , 0 : width : 2] = RED_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = GREEN_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = BLUE_POSITION
    return cfa_mask
    
def get_cfa_mask_rggb( height , width):
    cfa_mask = np.zeros((height , width))

    cfa_mask[0 : height: 2 , 0 : width : 2] = RED_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = BLUE_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = GREEN_POSITION
    return cfa_mask

########################################################################################################

def get_cfa_mask_bggr( height , width):
    cfa_mask = np.zeros((height , width))

    cfa_mask[0 : height: 2 , 0 : width : 2] = BLUE_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = RED_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = GREEN_POSITION
    return cfa_mask

########################################################################################################

def get_cfa_mask_grbg( height , width):
    cfa_mask = np.zeros((height , width))

    cfa_mask[0 : height: 2 , 0 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = GREEN_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = RED_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = BLUE_POSITION
    return cfa_mask

########################################################################################################

def get_cfa_mask_gbrg( height , width):
    cfa_mask = np.zeros((height , width))

    cfa_mask[0 : height: 2 , 0 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = GREEN_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = BLUE_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = RED_POSITION
    return cfa_mask

########################################################################################################

def get_individual_channels_rgbg(bayer_image):

    height = len(bayer_image)
    width = len(bayer_image[0])

    red = np.zeros((height , width))
    green = np.zeros((height , width))
    blue = np.zeros((height , width))

    red[0 : height : 2 , 0 : width : 2] = bayer_image[0 : height : 2 , 0 : width : 2]
    green[0 : height : 2 , 1 : width : 2] = bayer_image[0 : height : 2 , 1 : width : 2]
    blue[1 : height : 2 , 0 : width : 2] = bayer_image[1 : height : 2 , 0 : width : 2]
    green[1 : height : 2 , 1 : width : 2] = bayer_image[1 : height : 2 , 1 : width : 2]
    
    return red , green , blue
    
def get_individual_channels_grbg(bayer_image):
    height = len(bayer_image)
    width = len(bayer_image[0])

    red = np.zeros((height , width))
    green = np.zeros((height , width))
    blue = np.zeros((height , width))

    red[0 : height : 2 , 1 : width : 2] = bayer_image[0 : height : 2 , 1 : width : 2]
    green[0 : height : 2 , 0 : width : 2] = bayer_image[0 : height : 2 , 0 : width : 2]
    green[1 : height : 2 , 1 : width : 2] = bayer_image[1 : height : 2 , 1 : width : 2]
    blue[1 : height : 2 , 0 : width : 2] = bayer_image[1 : height : 2 , 0 : width : 2]
    
    return red , green , blue

########################################################################################################

def get_individual_channels_gbrg(bayer_image):
    height = len(bayer_image)
    width = len(bayer_image[0])

    red = np.zeros((height , width))
    green = np.zeros((height , width))
    blue = np.zeros((height , width))

    blue[0 : height : 2 , 1 : width : 2] = bayer_image[0 : height : 2 , 1 : width : 2]
    green[0 : height : 2 , 0 : width : 2] = bayer_image[0 : height : 2 , 0 : width : 2]
    green[1 : height : 2 , 1 : width : 2] = bayer_image[1 : height : 2 , 1 : width : 2]
    red[1 : height : 2 , 0 : width : 2] = bayer_image[1 : height : 2 , 0 : width : 2]
    
    return red , green , blue

########################################################################################################

def get_individual_channels_rggb(bayer_image):
    height = len(bayer_image)
    width = len(bayer_image[0])

    red = np.zeros((height , width))
    green = np.zeros((height , width))
    blue = np.zeros((height , width))

    green[0 : height : 2 , 1 : width : 2] = bayer_image[0 : height : 2 , 1 : width : 2]
    red[0 : height : 2 , 0 : width : 2] = bayer_image[0 : height : 2 , 0 : width : 2]
    blue[1 : height : 2 , 1 : width : 2] = bayer_image[1 : height : 2 , 1 : width : 2]
    green[1 : height : 2 , 0 : width : 2] = bayer_image[1 : height : 2 , 0 : width : 2]
    
    return red , green , blue

########################################################################################################

def get_individual_channels_bggr(bayer_image):
    height = len(bayer_image)
    width = len(bayer_image[0])

    red = np.zeros((height , width))
    green = np.zeros((height , width))
    blue = np.zeros((height , width))

    green[0 : height : 2 , 1 : width : 2] = bayer_image[0 : height : 2 , 1 : width : 2]
    blue[0 : height : 2 , 0 : width : 2] = bayer_image[0 : height : 2 , 0 : width : 2]
    red[1 : height : 2 , 1 : width : 2] = bayer_image[1 : height : 2 , 1 : width : 2]
    green[1 : height : 2 , 0 : width : 2] = bayer_image[1 : height : 2 , 0 : width : 2]
    
    return red , green , blue

########################################################################################################

def show_image(image):
    """
    Display a color image using OpenCV.
    """
    import cv2
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

########################################################################################################


########################################################################################################
