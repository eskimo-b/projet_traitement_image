import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import skimage 
from scipy.signal import convolve2d

'''
# Données
H = np.array([[1,2,3],[2,1,3],[1,3,1]])
# Points sources 
x1 = np.array([1,2,1,3])
y1 = np.array([3,3,2,2])
'''


x1 = [1, 3, 5, 7]
x2 = [2, 4, 6, 8]
y1 = [10, 20, 30, 40]
y2 = [15, 25, 35, 45]

H = homography_estimate(x1, y1, x2, y2)

(x2_fin, y2_fin) = homography_apply(H,x1,y1)

print(x2, x2_fin, y2, y2_fin)

        
        
        
                
        