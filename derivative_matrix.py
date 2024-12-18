import torch 
import numpy as np
def get_derivative_matrix(k):
    errors_matrix = np.load('res_diff_matrix8192.npy').astype(np.float32)
    scale = 7.0 / 255.0
    scaled_matrix = errors_matrix * (scale**2)
    derivative_matrix_y = np.zeros_like(scaled_matrix)
    derivative_matrix_x = np.zeros_like(scaled_matrix)
    for x in range(0,256): 
        if(x == k):
            for y in range(0,256):
                derivative_matrix_x[x][y] =  (scaled_matrix[x+k][y] - scaled_matrix[x][y])/(2 * scale)
        elif(x < k):
           for y in range(0,256):
                derivative_matrix_x[x][y] =  0   
        elif(x == 255-k):
            for y in range(0,256):
                derivative_matrix_x[x][y] = (scaled_matrix[x][y] - scaled_matrix[x-k][y])/(2 * scale)
        elif(x > 255-k):
            for y in range(0,256):
                derivative_matrix_x[x][y] = 0
        else:
            for y in range(0,256):
                if(scaled_matrix[x][y]<scaled_matrix[x-k][y] and scaled_matrix[x][y]<scaled_matrix[x+k][y]):
                    derivative_matrix_x[x][y]=0
                else:derivative_matrix_x[x][y] = (scaled_matrix[x+k][y] - scaled_matrix[x-k][y])/(2 * scale)
    for y in range(0,256): 
        if(y == k):
            for x in range(0,256):
                derivative_matrix_y[x][y] =  (scaled_matrix[x][y + k] - scaled_matrix[x][y])/(2 * scale)
        elif(y < k):
           for x in range(0,256):
                derivative_matrix_y[x][y] =  0 
        elif(y == 255-k):
            for x in range(0,256):
                derivative_matrix_y[x][y] = (scaled_matrix[x][y] - scaled_matrix[x][y - k])/(2 * scale)
        elif(y > 255-k):
            for x in range(0,256):
                derivative_matrix_y[x][y] = 0
        else:
            for x in range(0,256):
                if(scaled_matrix[x][y]<scaled_matrix[x][y-k] and scaled_matrix[x][y]<scaled_matrix[x][y+k]):
                    derivative_matrix_y[x][y] = 0
                else:derivative_matrix_y[x][y] = (scaled_matrix[x][y + k] - scaled_matrix[x][y-k])/(2 * scale)
    np.save('derivative_matrix_y.npy', derivative_matrix_y)
    np.save('derivative_matrix_x.npy', derivative_matrix_x)


get_derivative_matrix(1)

