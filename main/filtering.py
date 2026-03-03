##################################################################################################################

import numpy as np
import cv2
import os
import sys

##################################################################################################################

path = "../Data/data.png"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.convolution import *
from Utils.utils import *
# filtering

image = cv2.imread(path)
kernel = getGaussianKernel()

blue , green , red = image[: , : , 0] , image[: , : , 1] , image[: , : , 2]
out1 = conv2D(blue , kernel)
out2 = conv2D(green , kernel)
out3 = conv2D(red , kernel)

images = [out1 , out2 , out3]
titles = ["blue" , "green" , "red"]
show_image_grid(images, titles, max_cols=3)



