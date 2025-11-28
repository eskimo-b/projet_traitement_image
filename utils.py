import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def homography_apply(H,x1,y1):
    
    denom = H[2,0]*x1 + H[2,1]*y1 + H[2,2]

    x2 = (H[0,0]*x1 + H[0,1]*y1 + H[0,2]) / denom
    y2 = (H[1,0]*x1 + H[1,1]*y1 + H[1,2]) / denom

    return (x2, y2)


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

def homography_projection(I1, I2, x, y):
    h1,w1 = I1.shape
    h2, w2 = I2.shape
    x1 = np.array([0,w1-1,w1-1,0])
    y1 = np.array([0,0,h1-1,h1-1])
    H = homography_estimate(x,y,x1,y1)
    for j in range(h2): #y
        for i in range(w2): #x
            xs, ys = homography_apply(H, i, j)
            xs = int(np.round(xs))
            ys = int(np.round(ys))
            if ((0<= xs < w1) and (0 <= ys < h1)):
                I2[j, i] = I1[ys, xs]
    return I2

def homography_extraction(I1,x,y,w,h):
    I2 = np.zeros((h,w))
    x2 = np.array([0,w-1,w-1,0])
    y2 = np.array([0,0,h-1,h-1])
    H = homography_estimate(x2,y2,x,y)
    for j in range(h): #y
        for i in range(w): #x
            xs, ys = homography_apply(H, i, j)
            xs = int(np.round(xs))
            ys = int(np.round(ys))
            I2[j, i] = I1[ys, xs]
    return I2

def homography_cross_projection(I, x1, y1, x2, y2):
    n = 1024
    I_cp = I.copy()
    trampo = np.array((n, n))
    h,w = I.shape
    x3 = np.array([0,n-1,n-1,0])
    y3 = np.array([0,0,n-1,n-1])
    H1_tramp = homography_estimate(x1, y1, x3, y3)
    H2_tramp = homography_estimate(x2, y2, x3, y3)
    H1_2 = homography_estimate(x1, y1, x2, y2)
    H2_1 = homography_estimate(x2, y2, x1, y1)
    for j in range(h):
        for i in range(w):
            xs_1, ys_1 = homography_apply(H1_tramp, i, j)
            xs_2, ys_2 = homography_apply(H2_tramp, i, j)
            if ((0 <= xs_1 < w) and (0 <= ys_1 < h)):
                xs_1, ys_1 = homography_apply(H1_2, i, j)
                I[j, i] = I_cp[ys_1, xs_1]
            if ((0 <= xs_2 < w) and (0 <= ys_2 < h)):
                xs_2, ys_2 = homography_apply(H2_1, i, j)
                I[j, i] = I_cp[ys_2, xs_2]



