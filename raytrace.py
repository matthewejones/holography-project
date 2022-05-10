import numpy as np
import cv2
import json
#For giving arguments
import sys, os, getopt

import utils
from goldney.OSPR import OSPR

usage = """
Usage:
    python raytrace.py <inputimage> (from /images/ folder)
Flags:
    -s <angle>:         Apply raytracing for a tilted surface at an angle in degrees
    -c <radius>:        Apply raytracing for a cylinder with a radius curvature given in metres
    -R <distance>:      Specify distance of screen from projector
    -a:                 Specify beam angle
    -g:                 Generate hologram using ospr after generating the image
    -o <filename>:      Specify output filename
    -n:                 Don't save raytrace file
    --grid <width>:     Adjust size from grid and width of grid at projection distance instead of beam angle.
    --save <config>:    Save current configuration to file
    --config <config>:  Load configuration from file
"""


GRID_PROPORTION = (1014-264)/1280
beam_angle = utils.beam_angle
grid = False

def raytrace_slope(image, angle, R):
    #make 2 NxN arrays, one stores the total of the pixel values, one stores the number of pixels
    intensities = np.zeros(image.shape)
    quantities = np.zeros(image.shape, np.int16)
    if grid:
        pitch = grid / GRID_PROPORTION / image.shape[0]
    else:
        pitch = R * beam_angle / max(image.shape)

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

    #It would be better to instead of considering the beam angle, we consider the size of the projection at a given point.
    if grid:
        pitch = grid / GRID_PROPORTION / image.shape[0]
    else:
        pitch = R * beam_angle / max(image.shape)
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

    def parse_input(opts):
        global slope, cylinder, hologram, angle, r, R, output, save, beam_angle, grid
        for opt, arg in opts:
            if opt == '-s':
                slope = True
                angle = float(arg)
            elif opt == '-c':
                cylinder = True
                r = float(arg)
            elif opt == '-R':
                R = float(arg)
            elif opt == '-g':
                hologram = True
            elif opt == '-o':
                output = arg
            elif opt == '-n':
                save = False
            elif opt == '-a':
                beam_angle = arg
            elif opt == '--grid':
                grid = float(arg)

    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "s:c:R:o:gnha:", ['grid=', 'config=', 'save='])
    except:
        raise TypeError("Enter valid input image. Use -h for usage")
        sys.exit(2)

    
    #Check for configs
    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit(2)
        if opt == '--save':
            try:
                config_file = open('config/raytrace.json', 'r+')
            except FileNotFoundError:
                config_file = open('config/raytrace.json', 'w')
            finally:
                data = json.load(config_file)
                opts.remove((opt, arg))
                data[arg] = opts
                config_file.seek(0)
                json.dump(data, config_file, indent=4)
                config_file.close()
        if opt == '--config':
            with open('config/raytrace.json', 'r') as config_file:
                data = json.load(config_file)
                parse_input(data[arg])

    parse_input(opts)

    print("\n")
    print("#" * 65)
    print("#\tRaytrace Generator for Non-Uniform Projection\t\t#")
    print("#" * 65 + '\n')

    input_file = args[0]
    print("Processing ",input_file)

    name, extension = utils.split_filename(input_file)
    if not(extension):
        extension = 'bmp'
    
    filename = name

    
    
    input_image = cv2.imread('images/{}'.format(input_file), cv2.IMREAD_GRAYSCALE)

    if (slope):
        print("Raytracing...")
        raytrace = raytrace_slope(input_image, angle*np.pi/180, R)
        print("Raytracing complete")
        filename = output + ('-slope' if (slope and cylinder) else '') if output else "{}-slope-{}-{}".format(name, angle, R)
        if (save):
            cv2.imwrite('images/{}.{}'.format(filename, extension), raytrace)
            print("Saved as {}.{}".format(filename, extension))
    #Distance = 0.4375
    #r = 0.0225
    if (cylinder):
        print("Raytracing...")
        raytrace = raytrace_cylinder(input_image, r, R)
        print("Raytracing complete")
        filename = output + ('-cylinder' if (slope and cylinder) else '') if output else "{}-cylinder-{}-{}".format(name, r, R)
        if (save):
            cv2.imwrite('images/{}.{}'.format(filename, extension), raytrace)
            print("Saved as {}.{}".format(filename, extension))
    if (hologram):
        print("Generating Hologram")
        #From OSPR.py by Adam Goldney:
        transformed_image = utils.window_image_for_holo(raytrace)
        holo = OSPR(transformed_image, False)
        file = '{}-holo.{}'.format(filename, 'bmp')
        cv2.imwrite('holograms/{}'.format(file), holo)
        print("Saved as", file)



