import numpy as np
import cv2

import utils

def raytrace_slope(image, angle, R):
    #make 2 NxN arrays, one stores the total of the pixel values, one stores the number of pixels
    intensities = np.zeros(image.shape)
    quantities = np.zeros(image.shape, np.int16)



    pitch = R * utils.beam_angle / max(image.shape)
    width = image.shape[1]
    height = image.shape[0]
    for n in range(height):
        for m in range(width):
            u = (m - width/2)*pitch
            v = (height/2 - n)*pitch
            r = np.array([
                u * np.cos(angle),
                v,
                -R + u * np.sin(angle)
            ])
            r2 = -R/r[2] * r
            px2 = int(np.floor(r2[0]/pitch + width/2))
            py2 = int(np.floor(height/2 - r2[1]/pitch))
            #print(m, n, px2, py2)
            if (px2 < width and py2 < height):
                intensities[py2, px2] += image[n, m]
                quantities[py2, px2] += 1
    
    return np.divide(intensities, quantities, where=quantities > 0)

def raytrace_cylinder(image, radius, R):
    #make 2 NxN arrays, one stores the total of the pixel values, one stores the number of pixels
    intensities = np.zeros(image.shape)
    quantities = np.zeros(image.shape, np.int16)



    pitch = R * utils.beam_angle / max(image.shape)
    width = image.shape[1]
    height = image.shape[0]
    for n in range(height):
        for m in range(width):
            theta = (m - width/2)*pitch/radius
            v = (height/2 - n)*pitch
            r = np.array([
                radius * np.sin(theta),
                v,
                -R - radius * (1 - np.cos(theta))
            ])
            r2 = -R/r[2] * r
            px2 = int(np.floor(r2[0]/pitch + width/2))
            py2 = int(np.floor(height/2 - r2[1]/pitch))
            #print(m, n, px2, py2)
            if (px2 < width and py2 < height):
                intensities[py2, px2] += image[n, m]
                quantities[py2, px2] += 1
    
    return np.divide(intensities, quantities, where=quantities > 0)

if __name__ == "__main__":
    filename = "igrid.jpg"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    raytrace = raytrace_slope(image, 30*np.pi/180, 0.5)
    cv2.imwrite('30'+ filename, raytrace)



