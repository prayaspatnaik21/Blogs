import numpy as np

"""

   A method to improve Robustness of the Gray world Algorithm
   ==========================================================

   1. Color Constancy of human eyes can makes us recognize the true color of objects under complex illumination conditions.
   2. Image sensor records the real color of objects. 
   3. Image captured by image sensor depends not only on the color of the object but also the color temperature of external light source.
   4. White object under the low color temperature appears reddish , it is bluish in high color temperature.
   5. White balance algorithm corrects this color deviation dure to different color temperature of light source.

   Gray World Algorithm
   =====================

    1. Von kries Theory
        1. Color adaptation between different light sources can modeled as independent gain adjustments for each of the three color channels.
        2. Three cone types : 
            L - Long wavelength cones (sensitive to red)
            M - Medium wavelength cones (sensitive to green)
            S - Short wavelength cones (sensitive to blue)
        3. The transformation from one illuminant to another is:
            L_out = k_L * L_in
            M_out = k_M * M_in
            S_out = k_S * S_in
        4. L_in , M_in , S_in are the cone responses under the original illuminant.
        5. L_out , M_out , S_out are the cone responses under the new illuminant.
        6. k_L , k_M , k_S are the independent gain coefficients for each cone type.
        7. Why independent gain coefficients?
            1. Different spectral sensitivities : Each cone type responds different to the same light.
            2. Illuminant spectral power : Different light sources have different color temperatures.
            3. Neural Adaptation : The visual system adjusts each channel separately.
        8. These three gains coefficients are used to adjust an image under one light source to another one.

    2. Gray world Hypothesis
        1. The algorithm assumes that the average reflectance of the scene is achromatic (gray).
        2. For a scene with sufficient color variety:
            1. R(avg) = G(avg) = B(avg)
            2. R_avg = average value of all red pixels in the image.
            3. G_avg = average value of all green pixels in the image.
            4. B_avg = average value of all blue pixels in the image.
        3. Why this Assumption works?
            1. Natural scenes contain diverse colors : Most real world scenes have a balanced distribution of colors.
            2. Statistical color constancy : Over a large area , color casts tend to average out.
            3. Achromatic: if you average enough different colored objects , the result tends towards gray.
        4. Problem with this Assumption?
            1. The average value of three channels is not equal in non - standard light source.
            2. Means of three channels are greater or less than the grey value.
         
"""

def gray_world(image : np.ndarray):

    if image is None:
        raise ValueError("Image is None")

    if image.ndim != 3:
        raise ValueError("Image must be a 3D array")

    if image.shape[2] != 3:
        raise ValueError("Image must have exactly 3 channels")

    blue_channel = image[: , : , 0]
    green_channel = image[: , : , 1]
    red_channel = image[: , : , 2]

    blue_channel_average = np.mean(blue_channel)
    green_channel_average = np.mean(green_channel)
    red_channel_average = np.mean(red_channel)

    # Prevent division by zero
    if blue_channel_average == 0 or green_channel_average == 0 or red_channel_average == 0:
        raise ValueError("One or more channels have zero average - cannot apply gray world algorithm")

    gray = (blue_channel_average + green_channel_average + red_channel_average) / 3

    blue_gain_coefficient = gray / blue_channel_average
    green_gain_coefficient = gray / green_channel_average
    red_gain_coefficient = gray / red_channel_average

    # Convert to float for calculations to prevent overflow
    image_float = image.astype(np.float32)
    
    out = np.empty(image.shape , dtype = np.float32)

    out[: , : , 0] = blue_gain_coefficient * image_float[: , : , 0]
    out[: , : , 1] = green_gain_coefficient * image_float[: , : , 1]
    out[: , : , 2] = red_gain_coefficient * image_float[: , : , 2]

    # Clip values to valid range and convert back to original dtype
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
        out = np.clip(out, 0, max_val)
    
    return out.astype(image.dtype)
