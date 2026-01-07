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

def homography_cross_projection_1(I, x1, y1, x2, y2):
    h,w = 200,200
    # On extrait Q1 t Q2
    Q1 = homography_extraction(I, x1, y1, w, h)
    Q2 = homography_extraction(I, x2, y2, w, h)
    # On projete 
    r1 = homography_projection(Q1, I, x2, y2)
    r2 = homography_projection(Q2, r1, x1, y1)
    return r2


def homography_cross_projection(I, x1, y1, x2, y2):
    n = 1024
    I_cp = I.copy()
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
            if ((0 <= xs_1 < n) and (0 <= ys_1 < n)):
                xs_1, ys_1 = homography_apply(H1_2, i, j)
                xs_1 = int(np.round(xs_1))
                ys_1 = int(np.round(ys_1))
                I[j, i] = I_cp[ys_1, xs_1]
            if ((0 <= xs_2 < n) and (0 <= ys_2 < n)):
                xs_2, ys_2 = homography_apply(H2_1, i, j)
                xs_2 = int(np.round(xs_2))
                ys_2 = int(np.round(ys_2))
                I[j, i] = I_cp[ys_2, xs_2]
    return I

def ItoMIB(I):
    h,w = I.shape
    M = np.ones((h,w))
    B = np.array([0, 0, h-1, w-1])
    return M, I, B
    

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
    
    new_w = xmax - xmin + 1 # +1 car on veut les pixels entre xmin et xmax
    new_h = ymax - ymin + 1
    
    new_I = np.zeros((new_h,new_w))
    new_M = np.zeros(new_I.shape)
    
    x = np.array([0, new_w-1, new_w-1, 0])
    y = np.array([0, 0, new_h-1, new_h-1])
    
    new_I = homography_projection(I, new_I, x, y)
    
    new_M[new_I>0] = 1
                
    return new_M,new_I,new_B


def MIB_Fusion(MIB_tab): # MIB_tab : les MIB transofrmées par MIB_transform
    n = len(MIB_tab)
    # On trouve la taille de l'image finale
    bboxes = [B for (_, _, B) in MIB_tab]
    xmin = min(int(B[0]) for B in bboxes)
    ymin = min(int(B[1]) for B in bboxes)
    xmax = max(int(B[2]) for B in bboxes)
    ymax = max(int(B[3]) for B in bboxes)
    
    h = ymax - ymin + 1
    w = xmax - xmin + 1
    
    # Initialisations
    accum = np.zeros((h, w), dtype=float)   # somme des valeurs (des intensités)
    count = np.zeros((h, w), dtype=int)     # nombre de contributions
    M_tot = np.zeros((h, w), dtype=np.uint8)
    
    for j in range(h):
        for i in range(w):
            for k in range(n):
                M_src,I_src,B_src = MIB_tab[k] # on prend la MIB de l'image source
                h_src, w_src = I_src.shape 
                bx0,by0,bx1,by1 = B_src # on prend ses dimensions
                
                # offsets dans l'image globale
                off_x = bx0 - xmin   # colonnes
                off_y = by0 - ymin   # lignes
        
                # parcourir les pixels sources
                rows, cols = np.nonzero(M_src)
                accum[off_y + rows, off_x + cols] += I_src[rows, cols]
                count[off_y + rows, off_x + cols] += 1
                M_tot[off_y + rows, off_x + cols] = 1
                
    # calculer moyenne et construire I_out 
    I_out = np.zeros((h, w), dtype=float)
    mask = count > 0
    I_out[mask] = accum[mask] / count[mask]
     
    B_tot = np.array([xmin, ymin, xmax, ymax], dtype=int)
    return M_tot, I_out, B_tot
                 
     

    
    

