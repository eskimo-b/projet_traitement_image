import numpy as np
import matplotlib.pyplot as plt
from utils import homography_cross_projection_1

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