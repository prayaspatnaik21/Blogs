###############################################################################################################

from Demosaic.HighOrderInterpolation import HighOrderInterpolationDemosaic
import sys
import os
import numpy as np
import cv2
from AWB.grayWorld import gray_world

###############################################################################################################

# Add the parent directory to the path to access Utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Utils.utils import read_dng_image, get_bayer_pattern , show_image

# Test with synthetic data for now
import numpy as np
from Utils.utils import get_cfa_mask_rggb, get_individual_channels_rggb , read_dng_image , get_bayer_pattern , show_image
from Utils.constants import RED_POSITION, BLUE_POSITION, GREEN_POSITION

###############################################################################################################

if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), "Data", "nature.dng")
    raw_image = read_dng_image(raw_path)

    # Get Bayer pattern
    pattern, pattern_desc = get_bayer_pattern(raw_path)
    bgr_image = HighOrderInterpolationDemosaic(raw_image, pattern_desc )

    # Add white balance 
    bgr_image = gray_world(bgr_image)
    

    # Apply gamma correction for better brightness
    gamma = 0.05
    gamma_corrected = np.power(bgr_image,  gamma)
    
    # Convert back to 8-bit
    final_image = (gamma_corrected * 255).astype(np.uint8)
    
    show_image(final_image)
   
    
    