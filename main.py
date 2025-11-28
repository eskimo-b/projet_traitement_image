import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import skimage 
from scipy.signal import convolve2d


def  homography_estimate(x1, y1, x2, y2):
    A = np.zeros((8,8))
    B = np.zeros(8)
    for i in range(4):
        x1_i = x1[i]
        y1_i = y1[i]
        x2_i = x2[i]
        y2_i = y2[i]
        B[2*i] = x2_i
        B[2*i + 1] = y2_i
        l_x = [x1_i, y1_i, 1, 0, 0, 0, -x2_i*x1_i, -x2_i*y1_i]
        l_y = [0, 0, 0, x1_i, y1_i, 1, -x1_i*y2_i, -y1_i*y2_i]
        A[2*i, :] = l_x
        A[2*i + 1, :] = l_y
    
    H = np.linalg.solve(A, B)
    
    H = np.append(H, 1)
    H = np.reshape(H, (3, 3))
    return H

def homography_apply(H,x1,y1):
    
    x1 = np.array(x1)
    y1 = np.array(y1)
    
    denom = H[2,0]*x1 + H[2,1]*y1 + H[2,2]

    x2 = (H[0,0]*x1 + H[0,1]*y1 + H[0,2]) / denom
    y2 = (H[1,0]*x1 + H[1,1]*y1 + H[1,2]) / denom

    return (x2, y2)



x1 = [1, 3, 5, 7]
x2 = [2, 4, 6, 8]
y1 = [10, 20, 30, 40]
y2 = [15, 25, 35, 45]

H = homography_estimate(x1, y1, x2, y2)

(x2_fin, y2_fin) = homography_apply(H,x1,y1)

print(x2, x2_fin, y2, y2_fin)

        
        
        
                
        