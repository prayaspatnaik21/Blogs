###########################################################################################################################

import numpy as np

###########################################################################################################################

def addPadding(image , pad):
    rows , cols =image.shape[0] , image.shape[1]
    out_rows , out_cols = rows + 2 * pad , cols + 2 * pad
    out = np.zeros((out_rows , out_cols))

    out[pad : out_rows - pad, pad : out_cols - pad] = image
    
    return out

###########################################################################################################################

def conv2D(image , kernel , stride = 1 , isValid = False): 
    """

    Args:
        image (numpy array)
        kernel (numpy array)
    """

    rows , cols = image.shape[0] , image.shape[1]
    kernel_rows , kernel_cols = kernel.shape[0] , kernel.shape[1]

    pad = 0
    out_rows , out_cols = rows , cols
    if not isValid:
        pad = (kernel_rows - 1) // 2
        image = addPadding(image , pad)
    else:
        out_rows , out_cols = (rows - kernel_rows + 1) // stride , (cols - kernel_cols + 1) // stride
    
    out = np.zeros((out_rows , out_cols))

    # convolution
    row_id , col_id = 0 , 0
    #print(out_rows)
    while(row_id < out_rows):
        while(col_id < out_cols):
            sumConv = 0
            count = 0
            for curr_row_id in range(row_id , row_id + kernel_rows):
                for curr_col_id in range(col_id , col_id + kernel_cols):
                    sumConv += image[curr_row_id][curr_col_id] * kernel[curr_row_id - row_id][curr_col_id - col_id]
                    count += 1
            
            out[row_id][col_id] = sumConv // count
            col_id += stride
        row_id += stride
        col_id = 0
        #print(row_id)
    
    print(out)
    return out

###########################################################################################################################

def getGaussianKernel(kernel_size = 3 , sigma = 1):
    out = np.zeros((kernel_size , kernel_size))

    for row_id in range(kernel_size):
        for col_id in range(kernel_size):
            # gaussian equation = 1/2 * pi * sigma**2 * e power -(x**2 + y**2) / 2 sigma**2
            kernel_val = (1 / (2 * np.pi * sigma * sigma)) * np.exp(-(row_id**2 + col_id**2)/ (2 * sigma * sigma))
            out[row_id][col_id] = kernel_val
    print(out)
    return out

###########################################################################################################################

# data = np.array([[1,1,1],[1,1,1] , [1,1,1]])
# kernel = data
# conv2D(data , kernel)
# getGaussianKernel()