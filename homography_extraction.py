#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:19:55 2025

@author: meryem
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
            I2[j, i] = I1[ys, xs]
    return I2


points = []

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    points.append([x, y])
    
    plt.plot(event.xdata, event.ydata, "ro")
    plt.draw()
    
    if len(points) == 4:
       plt.close()

    print("\nPoints sélectionnés :", points)

I1 = Image.open('./img/forme_quadrangulaire.jpg')
I1_array = np.array(I1)
plt.figure("Sélection des 4 coins")
plt.imshow(I1)
plt.title("Cliquez 4 points dans l'ordre des coins")
cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)
plt.show(block=True)

x = [points[k][0] for k in range(len(points))]
y = [points[k][1] for k in range(len(points))]

w = 200
h = 350

result = homography_extraction(I1_array,x,y,w,h)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(I1)
plt.title('Image de base')
plt.subplot(1,2,2)
plt.imshow(result)
plt.title('Image extraite')