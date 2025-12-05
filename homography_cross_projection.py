#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 11:14:13 2025

@author: meryem
"""

import numpy as np
import matplotlib.pyplot as plt


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


def homography_extraction(I1,x,y,w,h):
    # on initialise I2 et les points x2 et y2 de l'image obtenue
    I2 = np.zeros((h,w)) 
    x2 = np.array([0,w-1,w-1,0]) 
    y2 = np.array([0,0,h-1,h-1])
    # on applique l'homographie entre l'image souhaitée et l'image de base 
    H = homography_estimate(x2,y2,x,y)
    for j in range(h): #y
        for i in range(w): #x
            xs, ys = homography_apply(H, i, j) # on applique 
            xs = int(np.round(xs)) # on caste en int pour bien reconstituer I2
            ys = int(np.round(ys))
            if 0 <= xs < I1.shape[1] and 0 <= ys < I1.shape[0]:
                I2[j, i] = I1[ys, xs]
    return I2


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


def homography_cross_projection_1(I, x1, y1, x2, y2):
    
    h,w = 200,200
    
    # On extrait Q1 t Q2
    Q1 = homography_extraction(I, x1, y1, w, h)
    Q2 = homography_extraction(I, x2, y2, w, h)

    # On projete 
    r1 = homography_projection(Q1, I, x2, y2)
    r2 = homography_projection(Q2, r1, x1, y1)
    
    return r2


points = []

# fonction faite par IA

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    points.append([x, y])
    
    plt.plot(event.xdata, event.ydata, "ro")
    plt.draw()
    
    if len(points) == 8:
       plt.close()

    print("\nPoints sélectionnés :", points)

img = plt.imread('./img/meb.jpg')
I1_array = np.array(img)
plt.figure("Sélection des 4 coins 2 fois")
plt.imshow(img)
plt.title("Cliquez 8 points dans l'ordre des coins")
cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)
plt.show(block=True)

x1 = [points[k][0] for k in range(len(points)//2)]
y1 = [points[k][1] for k in range(len(points)//2)]

x2 = [points[k][0] for k in range(len(points)//2, len(points))]
y2 = [points[k][1] for k in range(len(points)//2, len(points))]


result = homography_cross_projection_1(I1_array, x1, y1, x2, y2)
plt.imshow(result)