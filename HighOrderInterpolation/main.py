###############################################################################################################

import HighOrderInterpolation as hoi
import sys
import os
import numpy as np
###############################################################################################################

# Add the parent directory to the path to access Utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.utils import read_dng_image, get_bayer_pattern , show_image

###############################################################################################################

if __name__ == "__main__":
    raw_path = "../Data/nature3.dng"
    raw_image = read_dng_image(raw_path)
    
    print("Raw Image Shape: ", raw_image.shape)
    print("Height: ", raw_image.shape[0])
    print("Width: ", raw_image.shape[1])
    print(np.max(raw_image))
    # Get Bayer pattern
    pattern, pattern_desc = get_bayer_pattern(raw_path)
    print("Bayer Pattern: ", pattern)
    print("Pattern Description: ", pattern_desc)

    bgr_image = hoi.HighOrderInterpolationDemosaic(raw_image, pattern_desc)
    #print(bgr_image)
    print(bgr_image.shape)
    bgr_image = ((bgr_image/4095) * 255).astype(np.uint8)
    print(np.max(bgr_image))
    show_image(bgr_image)
    #print("BGR Image Shape: ", bgr_image.shape)
    
    