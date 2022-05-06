import numpy as np
import cv2
#For giving arguments
import sys, os, getopt

import utils
from goldney.OSPR import OSPR

usage = """
Usage:
    python raytrace.py <inputimage> (from /images/ folder)
Flags:
    -s <angle>:     Apply raytracing for a tilted surface at an angle in degrees
    -c <radius>:    Apply raytracing for a cylinder with a radius curvature given in metres
    -R <distance from projector to screen>
    -g:             Generate hologram using ospr after generating the image
    -o <filename>:  Specify output filename
    -n:             Don't save raytrace file
"""

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
    slope = cylinder = hologram = False
    save = True
    R = 0.5
    output = ''
    angle = 0
    r = 0.1

    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "s:c:R:o:gnh")
    except:
        raise TypeError("Enter valid input image. Use -h for usage")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit(2)
        elif opt == '-s':
            slope = True
            angle = float(arg)
        elif opt == '-c':
            cylinder == True
            r = float(arg)
        elif opt == '-R':
            R = float(arg)
        elif opt == '-g':
            hologram = True
        elif opt == '-o':
            output = arg
        elif opt == '-n':
            save = False

    print("\n")
    print("#" * 65)
    print("#\tRaytrace Generator for Non-Uniform Projection\t\t#")
    print("#" * 65)

    input_file = args[0]

    name, extension = utils.split_filename(input_file)
    if not(extension):
        extension = 'bmp'
    
    
    input_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    if (slope):
        raytrace = raytrace_slope(input_image, angle, R)
        if (save):
            filename = output + ('-slope' if (slope and cylinder) else '') if output else "{}-slope-{}-{]".format(name, angle, R)
            cv2.imwrite('images/{}.{}'.format(filename, extension), raytrace)
    #Distance = 0.4375
    #r = 0.0225
    if (cylinder):
        raytrace = raytrace_cylinder(input_image, r, R)
        if (save):
            filename = output + ('-cylinder' if (slope and cylinder) else '') if output else "{}-cylinder-{}-{]".format(name, r, R)
            cv2.imwrite('images/{}.{}'.format(filename, extension), raytrace)
    if (hologram):
        #From OSPR.py by Adam Goldney:
        transformed_image = utils.window_image_for_holo(raytrace)
        holo = OSPR(transformed_image, False)
        cv2.imwrite('holograms/{}-holo.{}'.format(filename, 'bmp'), holo)



