# Written for CMMPE by Adam Goldney
# Copyright 2020-2021

# Contains all accessory functions for hologram generation
import numpy as np
from scipy.optimize import curve_fit
import math
import cv2
import sys, getopt, os
import matplotlib.pyplot as plt

graph = cv2.imread('Gamma.png')

x = np.arange(0,graph.shape[1])
y = np.zeros_like(x)


graph2 = np.zeros([graph.shape[0], graph.shape[1]])
for i in range(0, graph.shape[0]):
    for j in range(0, graph.shape[1]):
        if np.dot(graph[i,j], [1,1,1]) > 75:
            y[j] = graph.shape[1] - i

y = y[y!=0]
x = np.arange(0,len(y))
xmax = np.max(x)
y = (y - np.min(y))/(np.max(y)- np.min(y))
x = np.linspace(0,1,len(y))

coeffs = np.polyfit(x, y, 3)
poly_fit_func = np.poly1d(coeffs)


print(coeffs)
y2 = poly_fit_func(x)



coeffs2 = np.polyfit(x, y2, 3)
poly2 = np.poly1d(coeffs2)
y3 = poly2(x)

y3 = x**(0.6)
#cv2.imshow('Graph',graph.astype(np.float64)/255)
#cv2.waitKey(0)
plt.plot(x,y2)
plt.plot(x, y3)
plt.show()
print(poly2)
