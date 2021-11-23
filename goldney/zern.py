from holo_projector import *

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

img = cv2.imread('Images/grid.jpg', cv2.IMREAD_GRAYSCALE)

r = 0.1
z0 = 0.3
type = 'polar'
no_slices = 10

imgs, z = slice_image(img, z0, type, r, no_slices)

print(z[-1])
cv2.imshow('Simulated Replay Field at z = {}'.format(z0),imgs[0]/255)
cv2.waitKey(0)

