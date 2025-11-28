from homography_extraction import homography_extraction
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
plt.imshow(result)

