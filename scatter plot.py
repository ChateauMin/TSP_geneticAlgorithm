import matplotlib.pyplot as plt
import numpy as np

plt.xlabel("k = searchPressure")
plt.ylabel("pop size")
plt.title("Correlation between k and pop size")
plt.axis([1.8, 3.2, -200, 2400])     # X, Y축의 범위: [xmin, xmax, ymin, ymax]

x1 = [2.0, 2.2, 2.3, 2.31, 2.4]
y1 = [200, 200, 200, 250, 230]
cost1 = 6381.72
color1 = 'Green'
area1 = (cost1/500)**3
for i in range(5):
    area4 = (cost1/1000)**3
    plt.scatter(x1[i], y1[i], s=area1, c=color1, alpha=0.6)

x2 = [2.2, 2.3, 2.4, 2.5, 2.55]
y2 = [200, 300, 400, 350, 400]
cost2 = 6001.13
color2 = 'Yellow'
area2 = (cost2/600)**3
for i in range(5):
    area4 = (cost2/1000)**3
    plt.scatter(x2[i], y2[i], s=area2, c=color2, alpha=0.6)

x3 = [2.45, 2.5, 2.5, 2.55, 2.6]
y3 = [1000, 1500, 2000, 2000, 2200]
cost3 = 5625.92
color3 = 'Purple'
area3 = (cost3/700)**3
for i in range(5):
    area4 = (cost3/1000)**3
    plt.scatter(x3[i], y3[i], s=area3, c=color3, alpha=0.6)

x4 = [2.65, 2.7, 2.7, 2.75, 2.75]
y4 = [900, 900, 1000, 950, 1000]
cost4 = 5208
color4 = 'Black'
for i in range(5):
    area4 = (cost4/1000)**3
    plt.scatter(x4[i], y4[i], s=area4, c=color4, alpha=0.6)

plt.show()



