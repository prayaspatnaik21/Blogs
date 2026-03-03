########################################################################################################

import rawpy
import numpy as np
from .constants import RED_POSITION , BLUE_POSITION , GREEN_POSITION
import matplotlib.pyplot as plt
import cv2
########################################################################################################

"""

1. Whenever you are reading a DNG file , use raw_image_visible to get the image data.
2. if you use raw.postprocess() to get the image data, it will convert the raw image to RGB image.

"""

########################################################################################################

def find_bit_depth(file_path):
    """
    Find the bit depth of a DNG image.
    
    Args:
        file_path (str): Path to the DNG file
        
    Returns:
        int: Bit depth of image
    """
    with rawpy.imread(file_path) as raw:
        bit_depth = raw.raw_image_visible.dtype.itemsize * 8
    return bit_depth

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

def get_cfa_mask_rggb( height , width):
    cfa_mask = np.zeros((height , width))
    
    cfa_mask[0 : height: 2 , 0 : width : 2] = RED_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = BLUE_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = GREEN_POSITION
    return cfa_mask
    
def get_cfa_mask_bggr( height , width):
    cfa_mask = np.zeros((height , width))
    
    cfa_mask[0 : height: 2 , 0 : width : 2] = BLUE_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = RED_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = GREEN_POSITION
    return cfa_mask
    
def get_cfa_mask_grbg( height , width):
    cfa_mask = np.zeros((height , width))
    
    cfa_mask[0 : height: 2 , 0 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = GREEN_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = RED_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = BLUE_POSITION
    return cfa_mask
    
def get_cfa_mask_gbrg( height , width):
    cfa_mask = np.zeros((height , width))
    
    cfa_mask[0 : height: 2 , 0 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = GREEN_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = BLUE_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = RED_POSITION
    return cfa_mask
    
def get_cfa_mask_rgbg( height , width):
    cfa_mask = np.zeros((height , width))
    
    cfa_mask[0 : height: 2 , 0 : width : 2] = RED_POSITION
    cfa_mask[1 : height : 2 , 1 : width : 2]  = GREEN_POSITION
    cfa_mask[0 : height : 2 , 1 : width : 2] = GREEN_POSITION
    cfa_mask[1 : height : 2, 0 : width : 2] = BLUE_POSITION
    return cfa_mask

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

def show_histogram(image , bit_depth):
    """
    Display the histogram of an image with RGB channels shown separately.
    Always uses 256 bins for consistent visualization.
    """
    import matplotlib.pyplot as plt
    image_bit_depth = np.iinfo(image.dtype).bits
    
    if image_bit_depth == bit_depth:
        # Always use 256 bins for consistent visualization
        bins = 256
        
        # Check if image is color (3 channels) or raw (1 channel)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Color image - split into RGB channels (assuming BGR format from OpenCV)
            blue_channel = image[:, :, 0]
            green_channel = image[:, :, 1] 
            red_channel = image[:, :, 2]
            
            plt.figure(figsize=(10, 6))
            
            # Plot histograms for each channel
            plt.hist(red_channel.ravel(), bins=bins, range=(0, 2**bit_depth), 
                    alpha=0.7, color='red', label='Red Channel')
            plt.hist(green_channel.ravel(), bins=bins, range=(0, 2**bit_depth), 
                    alpha=0.7, color='green', label='Green Channel')
            plt.hist(blue_channel.ravel(), bins=bins, range=(0, 2**bit_depth), 
                    alpha=0.7, color='blue', label='Blue Channel')
            
            plt.title(f"Histogram of Color Image with {bit_depth} bits (RGB Channels)")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Raw image - single channel
            plt.figure(figsize=(10, 6))
            plt.hist(image.ravel(), bins=bins, range=(0, 2**bit_depth), 
                    alpha=0.8, color='gray', edgecolor='black')
            plt.title(f"Histogram of Raw Image with {bit_depth} bits (Single Channel)")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
        
        # Set meaningful x-axis ticks for different bit depths
        if bit_depth == 16:
            max_val = 2**bit_depth
            ticks = [0, max_val//4, max_val//2, 3*max_val//4, max_val]
            tick_labels = ['0', '16K', '32K', '48K', '64K']
            plt.xticks(ticks, tick_labels)
        elif bit_depth == 12:
            max_val = 2**bit_depth
            ticks = [0, max_val//4, max_val//2, 3*max_val//4, max_val]
            tick_labels = ['0', '1K', '2K', '3K', '4K']
            plt.xticks(ticks, tick_labels)
        elif bit_depth == 8:
            plt.xticks([0, 64, 128, 192, 255])
        
        plt.show()
    else:
        print(f"The bit depth of the image is {image_bit_depth} , not {bit_depth}")


def show_raw_histogram(raw_image, bit_depth):
    """
    Display histogram specifically for raw (single-channel) images.
    """
    import matplotlib.pyplot as plt
    image_bit_depth = np.iinfo(raw_image.dtype).bits
    
    if image_bit_depth == bit_depth:
        bins = 256
        
        plt.figure(figsize=(10, 6))
        plt.hist(raw_image.ravel(), bins=bins, range=(0, 2**bit_depth), 
                alpha=0.8, color='darkgray', edgecolor='black')
        plt.title(f"Raw Image Histogram - {bit_depth} bits")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        # Set meaningful x-axis ticks for different bit depths
        if bit_depth == 16:
            max_val = 2**bit_depth
            ticks = [0, max_val//4, max_val//2, 3*max_val//4, max_val]
            tick_labels = ['0', '16K', '32K', '48K', '64K']
            plt.xticks(ticks, tick_labels)
        elif bit_depth == 12:
            max_val = 2**bit_depth
            ticks = [0, max_val//4, max_val//2, 3*max_val//4, max_val]
            tick_labels = ['0', '1K', '2K', '3K', '4K']
            plt.xticks(ticks, tick_labels)
        elif bit_depth == 8:
            plt.xticks([0, 64, 128, 192, 255])
        
        plt.show()
    else:
        print(f"The bit depth of the image is {image_bit_depth} , not {bit_depth}")

########################################################################################################
def showImage(image):
    if image.shape[0] > 0 and image.shape[1] > 0:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Image Shape Invalid : {image.shape[0]} , {image.shape[1]}")


def show_image_grid(images, titles, max_cols=3):
    """
    Display multiple images in a grid layout with titles.
    
    Args:
        images (list): List of images to display
        titles (list): List of titles for each image (same length as images)
        max_cols (int): Maximum number of columns per row (default: 3)
    """
    if len(images) != len(titles):
        raise ValueError("Number of images must match number of titles")
    
    if not images:
        print("No images to display")
        return
    
    # Calculate grid dimensions
    num_images = len(images)
    cols = min(max_cols, num_images)
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    # Get dimensions of first image to determine grid size
    first_img = images[0]
    if len(first_img.shape) == 3:
        img_height, img_width = first_img.shape[:2]
    else:
        img_height, img_width = first_img.shape
    
    # Create grid canvas
    grid_height = img_height * rows
    grid_width = img_width * cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place images in grid
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // cols
        col = idx % cols
        
        # Convert image to 3-channel if needed
        if len(img.shape) == 2:
            # Convert to uint8 first
            img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Keep as grayscale by duplicating the single channel to all 3 channels
            img_display = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        else:
            img_display = img.copy()
        
        # Ensure image is uint8 for display
        if img_display.dtype != np.uint8:
            img_display = cv2.normalize(img_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Calculate position in grid
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        # Place image
        grid[y_start:y_end, x_start:x_end] = img_display
        
        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        background_color = (0, 0, 0)  # Black background
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(title, font, font_scale, font_thickness)
        
        # Draw background rectangle for title
        title_y = y_start + 25
        cv2.rectangle(grid, (x_start, title_y - text_height - 5), 
                     (x_start + text_width + 10, title_y + 5), background_color, -1)
        
        # Draw title text
        cv2.putText(grid, title, (x_start + 5, title_y), font, font_scale, text_color, font_thickness)
    
    # Display the grid
    cv2.imshow("Image Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
########################################################################################################
