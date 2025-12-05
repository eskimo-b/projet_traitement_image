#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:15:05 2025

@author: meryem
"""
from utils import homography_apply, homography_projection
import numpy as np

def I_to_MIB(I):
    h,w = I.shape()
    M = np.ones(I)
    I_copy = np.copy(I)
    B = np.array([0,0,w-1,h-1])
    
    return M,I_copy,B
    
M,I,B = I_to_MIB()

def MIB_Transform(M,I,B,H):
    h,w = I.shape
    
    x1 = np.array([B[0],B[2]])
    y1 = np.array([B[1],B[3]])
    
    corners = np.array([[x1[0],y1[0]],[x1[0],y1[1]],[x1[1],y1[1]],[x1[1],y1[0]]])
    
    new_corners = []
    for x, y in corners:
        x2, y2 = homography_apply(H, x, y)
        new_corners.append([x2, y2])

    new_corners = np.array(new_corners)
    
    xmin = int(np.floor(new_corners[:,0].min()))
    xmax = int(np.ceil(new_corners[:,0].max()))
    ymin = int(np.floor(new_corners[:,1].min()))
    ymax = int(np.ceil(new_corners[:,1].max()))
    
    new_B = np.array([xmin,ymin,xmax,ymax])
    
    new_w = xmax - xmin + 1
    new_h = ymax - ymin + 1
    
    new_I = np.zeros((new_h,new_w))
    new_M = np.zeros(new_I)
    
    
    new_I = homography_projection(I, new_I, x, y)
    for i in range(new_h):
        for j in range(new_w):
            if (xmin<= i <=xmax) and (ymin<= j <=ymax) :
                new_I[j,i] = I[j,i]
                new_M[j,i] = 1
            else :
                new_I[j,i] = 0
                new_M[j,i] = 0
                
    return new_M,new_I,new_B
    