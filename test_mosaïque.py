#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:08:09 2025

@author: meryem
"""

from utils import ItoMIB, MIB_Fusion, MIB_Transform, homography_estimate
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


I = Image.open('./img/cameraman.tif')
I_array = np.array(I)

# Imagette 1 : coin en haut à gauche (50, 50), taille 180x200
I1 = I_array[50:230, 50:250]
# Imagette 2 : coin en haut à gauche (200, 100), taille 150x180
I2 = I_array[200:350, 100:280]


# Dimensions
h1, w1 = I1.shape
h2, w2 = I2.shape

# Coins de I1 (relatifs à I1)
h1, w1 = I1.shape
x1 = np.array([0, w1-1, w1-1, 0])
y1 = np.array([0, 0, h1-1, h1-1])

# Coins correspondants dans I2 (approximatifs)
x2 = x1 + 50
y2 = y1 + 50

# Calcul de l'homographie
H = homography_estimate(x1, y1, x2, y2)

M1, I1_mib, B1 = ItoMIB(I1)
M2, I2_mib, B2 = ItoMIB(I2)

M1_t, I1_t, B1_t = MIB_Transform(M1, I1_mib, B1, H)
M2_t, I2_t, B2_t = MIB_Transform(M2, I2_mib, B2, H)

MIB_tab = [(M1_t, I1_t, B1_t), (M2_t, I2_t, B2_t)]
M_tot, I_fusion, B_tot = MIB_Fusion(MIB_tab)

plt.figure()
plt.imshow(I, cmap='gray')
plt.title("Image test")
# -------------------
# Cadre imagette 1 (rouge)
x1, y1 = 50, 50       # coin supérieur gauche de I1
h1, w1 = I1.shape
plt.plot([x1, x1+w1], [y1, y1], 'r')        # haut
plt.plot([x1, x1+w1], [y1+h1, y1+h1], 'r')  # bas
plt.plot([x1, x1], [y1, y1+h1], 'r')        # gauche
plt.plot([x1+w1, x1+w1], [y1, y1+h1], 'r')  # droite

# Cadre imagette 2 (vert)
x2, y2 = 120, 120     # coin supérieur gauche de I2
h2, w2 = I2.shape
plt.plot([x2, x2+w2], [y2, y2], 'g')        # haut
plt.plot([x2, x2+w2], [y2+h2, y2+h2], 'g')  # bas
plt.plot([x2, x2], [y2, y2+h2], 'g')        # gauche
plt.plot([x2+w2, x2+w2], [y2, y2+h2], 'g')  # droite


plt.figure()
plt.imshow(I_fusion)
plt.show()

"""
# Affichage 
plt.figure()
plt.imshow(I_array)
plt.title("Image originale avec cadres")

# --- Cadre imagette 1 (rouge) ---
x1, y1 = 50, 50
plt.plot([x1, x1+w1], [y1, y1], 'r')              # haut
plt.plot([x1, x1+w1], [y1+h1, y1+h1], 'r')        # bas
plt.plot([x1, x1], [y1, y1+h1], 'r')              # gauche
plt.plot([x1+w1, x1+w1], [y1, y1+h1], 'r')        # droite

# --- Cadre imagette 2 (vert) ---
x2, y2 = 100, 200   # correspond aux indices de I2
plt.plot([x2, x2+w2], [y2, y2], 'g')              # haut
plt.plot([x2, x2+w2], [y2+h2, y2+h2], 'g')        # bas
plt.plot([x2, x2], [y2, y2+h2], 'g')              # gauche
plt.plot([x2+w2, x2+w2], [y2, y2+h2], 'g')        # droite

plt.show()
"""


