from utils import homography_projection
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    
    if len(points) == 4:
       plt.close()

    print("\nPoints sélectionnés :", points)


I1 = Image.open('./img/barbara.bmp')
I1 = np.array(I1)
plt.figure("Sélection des 4 coins")

I2 = Image.open('./img/cameraman.tif')
I2 = np.array(I2)
plt.imshow(I2)
cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)
plt.show(block=True)

x = [points[k][0] for k in range(len(points))]
y = [points[k][1] for k in range(len(points))]

I2 = homography_projection(I1, I2, x, y)

plt.imshow(I2)
