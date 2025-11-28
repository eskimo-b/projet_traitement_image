#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:08:49 2025

@author: meryem
"""

import numpy as np

# Données
H = np.array([[1,2,3],[2,1,3],[1,3,1]])
# Points sources 
x1 = np.array([1,2,1,3])
y1 = np.array([3,3,2,2])

def homography_apply(H,x1,y1):
    
    denom = H[2,0]*x1 + H[2,1]*y1 + H[2,2]

    x2 = (H[0,0]*x1 + H[0,1]*y1 + H[0,2]) / denom
    y2 = (H[1,0]*x1 + H[1,1]*y1 + H[1,2]) / denom

    return (x2, y2)

homography_apply(H, x1, y1)