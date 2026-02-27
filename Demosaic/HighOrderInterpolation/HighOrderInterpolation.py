import numpy as np
from enum import Enum
import os
import sys
import cv2
import math
import matplotlib.pyplot as plt
import time

# Add parent directories to path for Utils imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Utils.utils import (
    get_cfa_mask_rggb,
    get_cfa_mask_bggr,
    get_cfa_mask_grbg,
    get_cfa_mask_gbrg,
    get_cfa_mask_rgbg,
    get_individual_channels_rggb,
    get_individual_channels_bggr,
    get_individual_channels_grbg,
    get_individual_channels_gbrg,
    get_individual_channels_rgbg
)
from Utils.constants import (
    RED_POSITION,
    BLUE_POSITION, 
    GREEN_POSITION
)

"""

        Demosaicking Using High Order Interpolation Techniques with a weighted median for sharp color edge preservation
        ===============================================================================================================


        1. Three Algorithms for the estimation of green channel
            1. High Order Extrapolation
            2. High Order Interpolation
            3. Cubic Spline Interpolation.
        
        2. First Order Estimation of Red/Blue Channel 
        
"""

#########################################################################################################################################################

class GradientDirection(Enum):
    Vertical = 0
    Horizontal = 1

###############################################################################################################################################################################################

def corner_pixel_calculations(ne_left , ne_right , ne_top , ne_bottom):
    neighbors = [x for x in [ne_left, ne_right, ne_top, ne_bottom] if x is not None]
    interpolated_value = np.mean(neighbors) if neighbors else 0
    return interpolated_value

# High Order Extrapolation

def estimate_green_planes_high_order_extrapolation( bayer_image : np.ndarray,
                                                    height : int , 
                                                    width : int , 
                                                    cfa_mask : np.ndarray | None = None , 
                                                    red : np.ndarray | None = None , 
                                                    green : np.ndarray | None = None , 
                                                    blue : np.ndarray | None = None , 
                                                    corner_pixel_calculations = corner_pixel_calculations,
                                                    margin : int = 3,
                                                    out_dtype = np.float32,):
    """
        Algorithm
        =========

        1. Determine Four directional estimates of the Green channel.
        2. Assumption - High Spectral Correlations between the greena and red/blue pixels values within a local neighborhood.
        3. Missing Green value is surrounded by 4 known green values located in the left , right , top and bottom directions.
        4. Determine the missing values in the green plane by extrapolated in the four directions.
        5. The approximation is carried out using Taylor Series.
    """
    height , width = bayer_image.shape
    R = red.astype(np.float32 , copy = False)
    G = green.astype(np.float32 , copy = False)
    B = blue.astype(np.float32 , copy = False)

    g_estimate_left = G.astype(out_dtype , copy = True)
    g_estimate_right = G.astype(out_dtype , copy = True)
    g_estimate_top = G.astype(out_dtype , copy = True)
    g_estimate_bottom = G.astype(out_dtype , copy = True)

    is_red = (cfa_mask == RED_POSITION)
    is_blue = (cfa_mask == BLUE_POSITION)

    row_start  , row_end = margin , height - margin
    col_start , col_end = margin , width - margin

    if height >= 2 * margin + 1 and width >= 2 * margin + 1:

        rs = slice(row_start , row_end)
        cs = slice(col_start , col_end)

        def neighbor(arr , dx = 0 , dy = 0):
            return arr[row_start + dx : row_end + dx , col_start + dy : col_end + dy]

        red_interior = is_red[rs , cs]
        blue_interior = is_blue[rs , cs]
        
    ####################################################################################################################################
    # Estimating Green component  for blue pixels 
      
        np.putmask(
            g_estimate_left[rs , cs] , blue_interior,
            neighbor(G , 0 , -1) +
            0.75 * (neighbor(B , 0 , 0) - neighbor(B , 0 , -2)) -
            0.25 * (neighbor(G , 0 , -1) + neighbor(G , 0 , -3))
        )


        np.putmask(
            g_estimate_right[rs , cs] , blue_interior,
            neighbor(G , 0 , 1) +
            0.75 * (neighbor(B , 0 , 0) - neighbor(B , 0 , 2)) -
            0.25 * (neighbor(G , 0 , 1) - neighbor(G , 0 , 3))
        )

        np.putmask(
            g_estimate_top[rs , cs] , blue_interior, 
            neighbor(G , -1 , 0) +
            0.75 * (neighbor(B , 0 , 0) - neighbor(B , -2 , 0)) -
            0.25 * (neighbor(G , -1 , 0) - neighbor(G , -3 , 0))
        )

        np.putmask(
            g_estimate_bottom[rs , cs] , blue_interior, 
            neighbor(G , 1 , 0) +
            0.75 * (neighbor(B , 0 , 0) - neighbor(B , 2 , 0)) -
            0.25 * (neighbor(G , 1 , 0) - neighbor(G , 3 , 0))
        )

    ####################################################################################################################################
    # Estimating Green component  for red pixels
         
        np.putmask(
            g_estimate_left[rs , cs] , red_interior,
            neighbor(G , 0 , -1) +
            0.75 * (neighbor(B , 0 , 0) - neighbor(B , 0 , -2)) -
            0.25 * (neighbor(G , 0 , -1) + neighbor(G , 0 , -3))
        )


        np.putmask(
            g_estimate_right[rs , cs] , red_interior,
            neighbor(G , 0 , 1) +
            0.75 * (neighbor(B , 0 , 0) - neighbor(B , 0 , 2)) -
            0.25 * (neighbor(G , 0 , 1) - neighbor(G , 0 , 3))
        )

        np.putmask(
            g_estimate_top[rs , cs] , red_interior, 
            neighbor(G , -1 , 0) +
            0.75 * (neighbor(B , 0 , 0) - neighbor(B , -2 , 0)) -
            0.25 * (neighbor(G , -1 , 0) - neighbor(G , -3 , 0))
        )

        np.putmask(
            g_estimate_bottom[rs , cs] , red_interior, 
            neighbor(G , 1 , 0) +
            0.75 * (neighbor(B , 0 , 0) - neighbor(B , 2 , 0)) -
            0.25 * (neighbor(G , 1 , 0) - neighbor(G , 3 , 0))
        )

    ####################################################################################################################################

    if corner_pixel_calculations is not None:
        boundary_mask = np.ones((height , width) , dtype=bool)
        boundary_mask[row_start : row_end , col_start : col_end] = False
        red_blue_boundary = boundary_mask & (is_red | is_blue)

        coords = np.argwhere(red_blue_boundary)
        
        for row_id , col_id in coords:
            ne_left = G[row_id , col_id - 1] if col_id - 1 >= 0 else None
            ne_right = G[row_id , col_id + 1] if col_id + 1 < width else None
            ne_top = G[row_id - 1 , col_id] if row_id - 1 >= 0 else None
            ne_bottom = G[row_id + 1, col_id] if row_id  + 1 < height else None

            val = corner_pixel_calculations(ne_left , ne_right , ne_top , ne_bottom)
            
            g_estimate_left[row_id , col_id] = val
            g_estimate_right[row_id , col_id] = val
            g_estimate_top[row_id , col_id] = val
            g_estimate_bottom[row_id , col_id] = val

    return g_estimate_left , g_estimate_right , g_estimate_top , g_estimate_bottom

###############################################################################################################################################################################################

def estimate_green_planes_high_order_interpolation(
        bayer_image : np.ndarray,
        height : int , 
        width : int , 
        cfa_mask : np.ndarray | None = None , 
        red : np.ndarray | None = None , 
        green : np.ndarray | None = None , 
        blue : np.ndarray | None = None , 
        corner_pixel_calculations = corner_pixel_calculations , 
        margin : int = 4,
        out_dtype = np.float32,
):
    """

        Algorithm
        =========

        1. Determine Four directional estimates of the Green channel.
        2. Spectral Correlation referts to High Correlations between the greena and red/blue pixels values within a local neighborhood.
        3. Missing Green value is surrounded by 4 known green values located in the left , right , top and bottom directions.
        5. The approximation is carried out using Taylor Series.
        6. The different between interpolatin and Extrapolation is that interpolation include the adjacent green value on the other side of the missing sample
            in the interpolation process.
        7. Based on Presumption : Nearest known samples in a 2D plane to missing value contain the most accurate information about that missing sample.
        8. Corner Pixel Calculation are based on taking the mean of the surrounding pixels. (Can choose other methods as well.)

    """

    height , width = bayer_image.shape
    R = red.astype(np.float32 , copy = False)
    G = green.astype(np.float32 , copy = False)
    B = blue.astype(np.float32 , copy = False)

    g_estimate_left = G.astype(out_dtype , copy = True)
    g_estimate_right = G.astype(out_dtype , copy = True)
    g_estimate_top = G.astype(out_dtype , copy = True)
    g_estimate_bottom = G.astype(out_dtype , copy = True)

    is_red = (cfa_mask == RED_POSITION)
    is_blue = (cfa_mask == BLUE_POSITION)

    row_start  , row_end = margin , height - margin
    col_start , col_end = margin , width - margin
    
    if height >= 2 * margin + 1 and width >= 2 * margin + 1:

        rs = slice(row_start , row_end)
        cs = slice(col_start , col_end)

        def neighbor(arr , dx = 0 , dy = 0):
            return arr[row_start + dx : row_end + dx , col_start + dy : col_end + dy]

        red_interior = is_red[rs , cs]
        blue_interior = is_blue[rs , cs]

    ####################################################################################################################################

        # Estimating Green component  for blue pixels 
        np.putmask(
            g_estimate_left[rs , cs] , blue_interior,
            neighbor(G , 0 , -1) +
            0.5 * (neighbor(B , 0 , 0) - neighbor(B , 0 , -2)) +
            0.125 * (neighbor(G , 0 , 1) - 2 * neighbor(G , 0 , -1) + neighbor(G , 0 , -3))
        )

        np.putmask(
            g_estimate_right[rs , cs] , blue_interior,
            neighbor(G , 0 , 1) +
            0.5 * (neighbor(B , 0 , 0) - neighbor(B , 0 , 2)) +
            0.125 * (neighbor(G , 0 , -1) - 2 * neighbor(G , 0 , 1) + neighbor(G , 0 , 3))
        )

        np.putmask(
            g_estimate_top[rs , cs] , blue_interior, 
            neighbor(G , -1 , 0) +
            0.5 * (neighbor(B , 0 , 0) - neighbor(B , -2 , 0)) +
            0.125 * (neighbor(G , 1 , 0) - 2 * neighbor(G , -1 , 0) + neighbor(G , -3 , 0))
        )

        np.putmask(
            g_estimate_bottom[rs , cs] , blue_interior, 
            neighbor(G , 1 , 0) +
            0.5 * (neighbor(B , 0 , 0) - neighbor(B , 2 , 0)) +
            0.125 * (neighbor(G , -1 , 0) - 2 * neighbor(G , 1 , 0) + neighbor(G , 3 , 0))
        )

    ####################################################################################################################################

        # Estimating Green Component for Red Pixel
        np.putmask(
            g_estimate_left[rs , cs] , red_interior,
            neighbor(G , 0 , -1) +
            0.5 * (neighbor(R , 0 , 0) - neighbor(R , 0 , -2)) +
            0.125 * (neighbor(G , 0 , 1) - 2 * neighbor(G , 0 , -1) + neighbor(G , 0 , -3))
        )

        np.putmask(
            g_estimate_right[rs , cs] , red_interior,
            neighbor(G , 0 , 1) +
            0.5 * (neighbor(R , 0 , 0) - neighbor(R , 0 , 2)) +
            0.125 * (neighbor(G , 0 , -1) - 2 * neighbor(G , 0 , 1) + neighbor(G , 0 , 3))
        )

        np.putmask(
            g_estimate_top[rs , cs] , red_interior, 
            neighbor(G , -1 , 0) +
            0.5 * (neighbor(R , 0 , 0) - neighbor(R , -2 , 0)) +
            0.125 * (neighbor(G , 1 , 0) - 2 * neighbor(G , -1 , 0) + neighbor(G , -3 , 0))
        )

        np.putmask(
            g_estimate_bottom[rs , cs] , red_interior, 
            neighbor(G , 1 , 0) +
            0.5 * (neighbor(R , 0 , 0) - neighbor(R , 2 , 0)) +
            0.125 * (neighbor(G , -1 , 0) - 2 * neighbor(G , 1 , 0) + neighbor(G , 3 , 0))
        )

    ####################################################################################################################################

    if corner_pixel_calculations is not None:
        boundary_mask = np.ones((height , width) , dtype=bool)
        boundary_mask[row_start : row_end , col_start : col_end] = False
        red_blue_boundary = boundary_mask & (is_red | is_blue)

        coords = np.argwhere(red_blue_boundary)
        
        for row_id , col_id in coords:
            ne_left = G[row_id , col_id - 1] if col_id - 1 >= 0 else None
            ne_right = G[row_id , col_id + 1] if col_id + 1 < width else None
            ne_top = G[row_id - 1 , col_id] if row_id - 1 >= 0 else None
            ne_bottom = G[row_id + 1, col_id] if row_id  + 1 < height else None

            val = corner_pixel_calculations(ne_left , ne_right , ne_top , ne_bottom)
            
            g_estimate_left[row_id , col_id] = val
            g_estimate_right[row_id , col_id] = val
            g_estimate_top[row_id , col_id] = val
            g_estimate_bottom[row_id , col_id] = val

    return g_estimate_left , g_estimate_right , g_estimate_top , g_estimate_bottom

###############################################################################################################################################################################################

def blue_plane(cfa_mask : np.ndarray , 
                            interpolated_green_plane : np.ndarray ,
                            blue : np.ndarray ):
    
    """
        
        Algorithm
        =========

        1. Higher Order approximation is required for Green Plane only.
        2. First Order is adequate for red/blue plane.
        3. Calculating the candiates for Red pxiel in four direction for the green pixel and taking the mean( Direction : UP , DOWN , LEFT , RIGHT)
        4. Calculating the candidates for Blue pixel in four direction for the blue pixel and taking the mean.(Direction : UPPER LEFT , UPPER RIGHT , DOWN LEFT , DOWN RIGHT)

    """

    B = blue.astype(np.float32 , copy = True)
    G = interpolated_green_plane.astype(np.float32 , copy = False)

    # masks
    missing_blue = (blue == 0)
    at_green = (cfa_mask == GREEN_POSITION)
    at_red = (cfa_mask == RED_POSITION)

    Bp = np.pad(B , 1 , mode = "constant" , constant_values=0.0)
    Gp = np.pad(G , 1 , mode="constant" , constant_values=0.0)

    ##############################################################################################################################################
    # Calculation for Green Site
    
    B_up    = Bp[0:-2, 1:-1]; G_up    = Gp[0:-2, 1:-1]
    B_down  = Bp[2:  , 1:-1]; G_down  = Gp[2:  , 1:-1]
    B_left  = Bp[1:-1, 0:-2]; G_left  = Gp[1:-1, 0:-2]
    B_right = Bp[1:-1, 2:  ]; G_right = Gp[1:-1, 2:  ]

    cand_up    = B_up    + (G - G_up)
    cand_down  = B_down  + (G - G_down)
    cand_left  = B_left  + (G - G_left)
    cand_right = B_right + (G - G_right)

    valid_up    = (B_up    != 0) & (G_up    != 0)
    valid_down  = (B_down  != 0) & (G_down  != 0)
    valid_left  = (B_left  != 0) & (G_left  != 0)
    valid_right = (B_right != 0) & (G_right != 0)

    sum_green_sites = (
        np.where(valid_up,    cand_up,    0.0) +
        np.where(valid_down,  cand_down,  0.0) +
        np.where(valid_left,  cand_left,  0.0) +
        np.where(valid_right, cand_right, 0.0)
    )
    cnt_green_sites = (
        valid_up.astype(np.int32) +
        valid_down.astype(np.int32) +
        valid_left.astype(np.int32) +
        valid_right.astype(np.int32)
    )

    mean_green_sites = np.divide(
        sum_green_sites, cnt_green_sites,
        out=np.zeros_like(sum_green_sites, dtype=np.float32),
        where=(cnt_green_sites > 0)
    )

    ##############################################################################################################################################
    # Calculation for Red Sites
    
    B_ul, G_ul = Bp[0:-2, 0:-2], Gp[0:-2, 0:-2]  # up-left
    B_ur, G_ur = Bp[0:-2, 2:  ], Gp[0:-2, 2:  ]  # up-right
    B_dl, G_dl = Bp[2:  , 0:-2], Gp[2:  , 0:-2]  # down-left
    B_dr, G_dr = Bp[2:  , 2:  ], Gp[2:  , 2:  ]  # down-right

    cand_ul = B_ul + (G - G_ul)
    cand_ur = B_ur + (G - G_ur)
    cand_dl = B_dl + (G - G_dl)
    cand_dr = B_dr + (G - G_dr)

    valid_ul = (B_ul != 0) & (G_ul != 0)
    valid_ur = (B_ur != 0) & (G_ur != 0)
    valid_dl = (B_dl != 0) & (G_dl != 0)
    valid_dr = (B_dr != 0) & (G_dr != 0)

    sum_red_sites = (
        np.where(valid_ul, cand_ul, 0.0) +
        np.where(valid_ur, cand_ur, 0.0) +
        np.where(valid_dl, cand_dl, 0.0) +
        np.where(valid_dr, cand_dr, 0.0)
    )
    cnt_red_sites = (
        valid_ul.astype(np.int32) +
        valid_ur.astype(np.int32) +
        valid_dl.astype(np.int32) +
        valid_dr.astype(np.int32)
    )
    mean_red_sites = np.divide(
        sum_red_sites, cnt_red_sites,
        out=np.zeros_like(sum_red_sites, dtype=np.float32),
        where=(cnt_red_sites > 0)
    )
    
    ##############################################################################################################################################
    
    write_green_sites = missing_blue & at_green & (cnt_green_sites > 0)
    write_blue_sites  = missing_blue & at_red  & (cnt_red_sites  > 0)

    B[write_green_sites] = mean_green_sites[write_green_sites]
    B[write_blue_sites]  = mean_red_sites [write_blue_sites]

    return B

###############################################################################################################################################################################################

def red_plane(cfa_mask : np.ndarray , 
                            interpolated_green_plane : np.ndarray ,
                            red : np.ndarray):
    """
        
        Algorithm
        =========

        1. Higher Order approximation is required for Green Plane only.
        2. First Order is adequate for red/blue plane.
        3. Calculating the candiates for Red pxiel in four direction for the green pixel and taking the mean( Direction : UP , DOWN , LEFT , RIGHT)
        4. Calculating the candidates for Blue pixel in four direction for the blue pixel and taking the mean.(Direction : UPPER LEFT , UPPER RIGHT , DOWN LEFT , DOWN RIGHT)

    """
    R = red.astype(np.float32 , copy = True)
    G = interpolated_green_plane.astype(np.float32 , copy = False)

    # masks
    missing_red = (red == 0)
    at_green = (cfa_mask == GREEN_POSITION)
    at_blue = (cfa_mask == BLUE_POSITION)

    Rp = np.pad(R , 1 , mode = "constant" , constant_values=0.0)
    Gp = np.pad(G , 1 , mode="constant" , constant_values=0.0)

    ##############################################################################################################################################
    # Calculation for Green Sites

    R_up    = Rp[0:-2, 1:-1]; G_up    = Gp[0:-2, 1:-1]
    R_down  = Rp[2:  , 1:-1]; G_down  = Gp[2:  , 1:-1]
    R_left  = Rp[1:-1, 0:-2]; G_left  = Gp[1:-1, 0:-2]
    R_right = Rp[1:-1, 2:  ]; G_right = Gp[1:-1, 2:  ]

    cand_up    = R_up    + (G - G_up)
    cand_down  = R_down  + (G - G_down)
    cand_left  = R_left  + (G - G_left)
    cand_right = R_right + (G - G_right)

    valid_up    = (R_up    != 0) & (G_up    != 0)
    valid_down  = (R_down  != 0) & (G_down  != 0)
    valid_left  = (R_left  != 0) & (G_left  != 0)
    valid_right = (R_right != 0) & (G_right != 0)

    sum_green_sites = (
        np.where(valid_up,    cand_up,    0.0) +
        np.where(valid_down,  cand_down,  0.0) +
        np.where(valid_left,  cand_left,  0.0) +
        np.where(valid_right, cand_right, 0.0)
    )
    cnt_green_sites = (
        valid_up.astype(np.int32) +
        valid_down.astype(np.int32) +
        valid_left.astype(np.int32) +
        valid_right.astype(np.int32)
    )

    mean_green_sites = np.divide(
        sum_green_sites, cnt_green_sites,
        out=np.zeros_like(sum_green_sites, dtype=np.float32),
        where=(cnt_green_sites > 0)
    )

    ##############################################################################################################################################

    # Calculation for Blue Sites
    R_ul, G_ul = Rp[0:-2, 0:-2], Gp[0:-2, 0:-2]  # up-left
    R_ur, G_ur = Rp[0:-2, 2:  ], Gp[0:-2, 2:  ]  # up-right
    R_dl, G_dl = Rp[2:  , 0:-2], Gp[2:  , 0:-2]  # down-left
    R_dr, G_dr = Rp[2:  , 2:  ], Gp[2:  , 2:  ]  # down-right

    cand_ul = R_ul + (G - G_ul)
    cand_ur = R_ur + (G - G_ur)
    cand_dl = R_dl + (G - G_dl)
    cand_dr = R_dr + (G - G_dr)

    valid_ul = (R_ul != 0) & (G_ul != 0)
    valid_ur = (R_ur != 0) & (G_ur != 0)
    valid_dl = (R_dl != 0) & (G_dl != 0)
    valid_dr = (R_dr != 0) & (G_dr != 0)

    sum_blue_sites = (
        np.where(valid_ul, cand_ul, 0.0) +
        np.where(valid_ur, cand_ur, 0.0) +
        np.where(valid_dl, cand_dl, 0.0) +
        np.where(valid_dr, cand_dr, 0.0)
    )
    cnt_blue_sites = (
        valid_ul.astype(np.int32) +
        valid_ur.astype(np.int32) +
        valid_dl.astype(np.int32) +
        valid_dr.astype(np.int32)
    )
    mean_blue_sites = np.divide(
        sum_blue_sites, cnt_blue_sites,
        out=np.zeros_like(sum_blue_sites, dtype=np.float32),
        where=(cnt_blue_sites > 0)
    )
    
    ##############################################################################################################################################
    
    write_green_sites = missing_red & at_green & (cnt_green_sites > 0)
    write_blue_sites  = missing_red & at_blue  & (cnt_blue_sites  > 0)

    R[write_green_sites] = mean_green_sites[write_green_sites]
    R[write_blue_sites]  = mean_blue_sites [write_blue_sites]

    return R

###############################################################################################################################################################################################

def weighted_median_green_plane(green : np.ndarray | None = None , 
                                g_estimated_left : np.ndarray | None = None , 
                                g_estimated_right : np.ndarray | None = None , 
                                g_estimated_top : np.ndarray | None = None, 
                                g_estimated_bottom : np.ndarray | None = None, 
                                edge_orientation_map : np.ndarray | None = None,
                                weights_horizontal= (2 , 2 , 1 , 1),
                                weights_vertical = (1 , 1 , 2 , 2)):
    
    """
        Algorithm
        =========

        1. Applies a weighted median filter to estimate missing green values in a Bayer pattern image.
        2. Uses an edge orientation map to classify pixels and select appropriate filter weights (horizontal or vertical).
        3. Stacks green estimates from neighboring pixels (left, right, top, bottom).
        4. Sorts the estimates and their corresponding weights.
        5. Computes the cumulative sum of weights to find the weighted median positions.
        6. Calculates the weighted median value for each pixel.
        7. Replaces missing green values with the computed weighted median, preserving existing green values.
    """
    
    non_green = (green == 0)
    estimates = np.stack([g_estimated_left , g_estimated_right , g_estimated_top , g_estimated_bottom] , axis = -1).astype(np.float32 , copy =False)

    wH = np.asarray(weights_horizontal , dtype = np.int32)
    wV = np.asarray(weights_vertical , dtype = np.int32)

    is_horizontal = (edge_orientation_map ==  GradientDirection.Horizontal)
    weights = np.where(is_horizontal[... , None] , wH , wV)

    order = np.argsort(estimates , axis = -1 , kind = "Stable")
    values_sorted = np.take_along_axis(estimates , order , axis = -1)
    weights_sorted = np.take_along_axis(estimates , order , axis = -1)

    cum_weight =  np.cumsum(weights_sorted , axis = -1)
    total_weight = cum_weight[... , -1]

    m1 = (total_weight + 1) // 2
    m2 = (total_weight + 2) // 2

    pos1 = (cum_weight >= m1[..., None]).argmax(axis=-1)
    pos2 = (cum_weight >= m2[..., None]).argmax(axis=-1)

    mid1 = np.take_along_axis(values_sorted, pos1[..., None], axis=-1)[..., 0]
    mid2 = np.take_along_axis(values_sorted, pos2[..., None], axis=-1)[..., 0]
    weighted_median = 0.5 * (mid1 + mid2)

    out = green.astype(weighted_median.dtype, copy=True)
    out = np.where(non_green, weighted_median, out)
  
    return out

###############################################################################################################################################################################################
    
def green_plane(bayer_image , edge_orientation_map ,cfa_mask , red , green , blue ,  height , width , estimation = "interpolation"):
    
    g_estimated_left , g_estimated_right , g_estimated_top , g_estimated_bottom = None , None , None , None

    if estimation == "Extrapolation":
        g_estimated_left , g_estimated_right , g_estimated_top , g_estimated_bottom = estimate_green_planes_high_order_extrapolation(bayer_image ,height , width , cfa_mask , red , green , blue)
    else:    
        g_estimated_left , g_estimated_right , g_estimated_top , g_estimated_bottom = estimate_green_planes_high_order_interpolation(bayer_image ,height , width , cfa_mask , red , green , blue)

    green = weighted_median_green_plane(green , g_estimated_left , 
                                g_estimated_right  , 
                                g_estimated_top , 
                                g_estimated_bottom , 
                                edge_orientation_map ,
                                weights_horizontal= (2 , 2 , 1 , 1),
                                weights_vertical = (1 , 1 , 2 , 2))
    return green


def calculate_edge_orientation_map(
    bayer_image: np.ndarray,
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    cfa_mask: np.ndarray
) -> np.ndarray:
    
    """
        Algorithm
        =========

        1. Calculates an edge orientation map for a Bayer image using color channel gradients.
        2. For each interior pixel (excluding a 2-pixel border):
            1. Identify the CFA (Color Filter Array) position: red, green, or blue.
            2. For red and blue sites:
                1. Compute vertical and horizontal gradients using the green channel.
                2. Mark as "vertical edge" if the vertical gradient is less than the horizontal gradient.
        3. For green sites:
            1. Determine if the green pixel is on a "red row" or "blue row" by checking horizontal neighbors.
            2. For green on red row:
                1. Compute vertical and horizontal gradients using the red channel.
                2. Mark as "vertical edge" if the vertical gradient is less than the horizontal gradient.
            3. For green on blue row:
                a. Compute vertical and horizontal gradients using the blue channel.
                b. Mark as "vertical edge" if the vertical gradient is less than the horizontal gradient.
        4. The resulting edge orientation map is a boolean array indicating vertical edge orientation at each pixel
    """
    height, width = bayer_image.shape
    edge_orientation_map = np.zeros((height, width), dtype=bool)

    # avoid borders because we look 1 pixel around
    start_row, start_col = 2, 2
    end_row, end_col     = height - 2, width - 2
    rs, cs = slice(start_row, end_row), slice(start_col, end_col)

    cfa_interior = cfa_mask[rs, cs]
    is_red   = (cfa_interior == RED_POSITION)
    is_green = (cfa_interior == GREEN_POSITION)
    is_blue  = (cfa_interior == BLUE_POSITION)

    green_up    = green[start_row - 1 : end_row - 1, start_col     : end_col    ]
    green_down  = green[start_row + 1 : end_row + 1, start_col     : end_col    ]
    green_left  = green[start_row     : end_row    , start_col - 1 : end_col - 1]
    green_right = green[start_row     : end_row    , start_col + 1 : end_col + 1]

    red_up    = red[start_row - 1 : end_row - 1, start_col     : end_col    ]
    red_down  = red[start_row + 1 : end_row + 1, start_col     : end_col    ]
    red_left  = red[start_row     : end_row    , start_col - 1 : end_col - 1]
    red_right = red[start_row     : end_row    , start_col + 1 : end_col + 1]

    blue_up    = blue[start_row - 1 : end_row - 1, start_col     : end_col    ]
    blue_down  = blue[start_row + 1 : end_row + 1, start_col     : end_col    ]
    blue_left  = blue[start_row     : end_row    , start_col - 1 : end_col - 1]
    blue_right = blue[start_row     : end_row    , start_col + 1 : end_col + 1]

    mask_rb = is_red | is_blue
    green_vertical_gradient   = np.abs(green_up - green_down)
    green_horizontal_gradient = np.abs(green_left - green_right)
    edge_rb = green_vertical_gradient < green_horizontal_gradient
    edge_rb = edge_rb & mask_rb  

    left_is_red  = np.zeros_like(is_red, dtype=bool)
    right_is_red = np.zeros_like(is_red, dtype=bool)
    left_is_blue  = np.zeros_like(is_blue, dtype=bool)
    right_is_blue = np.zeros_like(is_blue, dtype=bool)

    left_is_red[:, 1:]   = (cfa_interior[:, :-1] == RED_POSITION)
    left_is_blue[:, 1:]  = (cfa_interior[:, :-1] == BLUE_POSITION)

    right_is_red[:, :-1]  = (cfa_interior[:, 1:] == RED_POSITION)
    right_is_blue[:, :-1] = (cfa_interior[:, 1:] == BLUE_POSITION)

    green_on_red_row  = is_green & (left_is_red | right_is_red)
    green_on_blue_row = is_green & (left_is_blue | right_is_blue)

    red_vertical_gradient   = np.abs(red_up - red_down)
    red_horizontal_gradient = np.abs(red_left - red_right)
    edge_g_redrow = red_vertical_gradient < red_horizontal_gradient
    edge_g_redrow = edge_g_redrow & green_on_red_row

    blue_vertical_gradient   = np.abs(blue_up - blue_down)
    blue_horizontal_gradient = np.abs(blue_left - blue_right)
    edge_g_bluerow = blue_vertical_gradient < blue_horizontal_gradient
    edge_g_bluerow = edge_g_bluerow & green_on_blue_row

    interior_edge = np.zeros_like(cfa_interior, dtype=bool)
    interior_edge[mask_rb]            = edge_rb[mask_rb]
    interior_edge[green_on_red_row]   = edge_g_redrow[green_on_red_row]
    interior_edge[green_on_blue_row]  = edge_g_bluerow[green_on_blue_row]

    edge_orientation_map[rs, cs] = interior_edge
    return edge_orientation_map  

###############################################################################################################################################################################################

def HighOrderInterpolationDemosaic(bayer_image , bayer_pattern , input_bit_depth = 12):
    """
        1. Demosaic Algorithm using High Order Interpoaltion and Extrapolation.
        2. Create RGB Image of same input bit depth.
        3. Max Input Bit Depth should be 16.
    """
    height = len(bayer_image)
    width  = len(bayer_image[0])

    data_type = None

    data_type = np.uint16 if input_bit_depth > 8 else np.uint8
    

    ###############################################################################################################################################
    
    cfa_mask = None
    red , green , blue = None , None , None 
    if bayer_pattern ==b'RGGB':
        cfa_mask = get_cfa_mask_rggb(height , width)
        red , green , blue = get_individual_channels_rggb(bayer_image)
    elif bayer_pattern ==b'BGGR':
        cfa_mask = get_cfa_mask_bggr(height , width)
        red , green , blue = get_individual_channels_bggr(bayer_image)
    elif bayer_pattern ==b'GBRG':
        cfa_mask = get_cfa_mask_gbrg(height , width)
        red , green , blue = get_individual_channels_gbrg(bayer_image)
    elif bayer_pattern ==b'RGBG':
        cfa_mask = get_cfa_mask_rgbg(height , width)
        red , green , blue = get_individual_channels_rgbg(bayer_image)
    else:
        # Default to GRBG pattern
        cfa_mask = get_cfa_mask_grbg(height , width)
        red , green , blue = get_individual_channels_grbg(bayer_image)
    
    ###############################################################################################################################################
    
    
    edge_orientation_map = calculate_edge_orientation_map(bayer_image , red , green , blue , cfa_mask)

    green = green_plane(bayer_image ,edge_orientation_map , cfa_mask , red , green , blue ,  height , width)
    red = red_plane(cfa_mask , green , red)
    blue = blue_plane(cfa_mask  , green , blue)
    
    bgr_image = np.stack((blue , green , red ), axis=-1)
    max_value = (1 << input_bit_depth) - 1
    bgr_image = np.clip(bgr_image , 0 , max_value).astype(data_type , copy = False)
    return bgr_image
    
##################################################################################################################################################