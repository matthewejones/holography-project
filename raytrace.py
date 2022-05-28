import numpy as np
import cv2
import json
from numba import jit
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
    -b:                 Take into account beam angle adjustment
    --grid <width>:     Adjust size from grid and width of grid at projection distance instead of beam angle.
    --save <config>:    Save current configuration to file
    --config <config>:  Load configuration from file
    --gamma:            Perform gamma correction
    --shade <i>:        Perform shading:
                            i = 1: linear vertical
                            i = 2: linear vertical inverted
                            i = 3: Horizontal linear symmetric
                            i = 4: Horizontal linear symmetric inverted
    -i:                 Apply transformation for a concave surface

"""


GRID_PROPORTION = (1014-264)/1280
beam_angle = utils.beam_angle
grid = False

BEAM_ANGLE_ADJUSTED = 0.102
R_ADJUSTMENT = 0.0196

def raytrace_slope(image, angle, R):
    intensities, quantities = _raytrace_slope(image, angle, R)
    
    return np.divide(intensities, quantities, where=quantities > 0)

@jit(nopython = True, cache = True)
def _raytrace_slope(image, angle, R):
    global grid
    #make 2 NxN arrays, one stores the total of the pixel values, one stores the number of pixels
    intensities = np.zeros(image.shape)
    quantities = np.zeros(image.shape, np.int16)
    if grid:
        pitch = grid / GRID_PROPORTION / image.shape[0]
    else:
        pitch = R * beam_angle / max(image.shape)


    #Can this be vectorised?? 
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
        
    return intensities, quantities


def raytrace_cylinder(image, radius, R):
    intensities, quantities = _raytrace_cylinder(image, radius, R)
    
    return np.divide(intensities, quantities, where=quantities > 0)

@jit(nopython = True, cache=True)
def _raytrace_cylinder(image, radius, R):
    global grid
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
            if(-np.pi/2 <= theta <= np.pi/2):
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
    
    return intensities, quantities

if __name__ == "__main__":
    slope = cylinder = hologram = False
    save = True
    R = 0.5
    output = ''
    angle = 0
    r = 0.1
    adjust = False
    gamma = False
    shade = False
    invert = False

    def parse_input(opts):
        global slope, cylinder, hologram, angle, r, R, output, save, beam_angle, grid, adjust, gamma, shade, invert
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
            elif opt == '-b':
                adjust = True
            elif opt == '--grid':
                grid = float(arg)
            elif opt == '--gamma':
                gamma = True
            elif opt == '--shade':
                shade = int(arg)
            elif opt == '-i':
                invert = True

    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "s:c:R:o:gnha:bi", ['grid=', 'config=', 'save=', 'gamma', 'shade='])
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

    if shade:
        input_image = utils.shade(input_image, shade)

    if invert:
        r = -r
        R += r

    if (adjust):
        print("Performing beam angle adjustment")
        beam_angle = BEAM_ANGLE_ADJUSTED
        R -= R_ADJUSTMENT

    if (slope):
        print("Raytracing...")
        raytrace = raytrace_slope(input_image, angle*np.pi/180, R)
        print("Raytracing complete")
        filename = output + ('-slope' if (slope and cylinder) else '') if output else "{}-slope-{}-{}{}".format(name, angle, R, shade if shade else '')
        if (save):
            cv2.imwrite('images/{}.{}'.format(filename, extension), raytrace)
            print("Saved as {}.{}".format(filename, extension))
    #Distance = 0.4375
    #r = 0.0225
    if (cylinder):
        print("Raytracing...")
        raytrace = raytrace_cylinder(input_image, r, R)
        print("Raytracing complete")
        filename = output + ('-cylinder' if (slope and cylinder) else '') if output else "{}-cylinder-{}-{}{}".format(name, r, R, shade if shade else '')
        if (save):
            cv2.imwrite('images/{}.{}'.format(filename, extension), raytrace)
            print("Saved as {}.{}".format(filename, extension))
    if (hologram):
        print("Generating Hologram")
        #From OSPR.py by Adam Goldney:
        if gamma == True:
            input_image = utils.gamma_correct(input_image)
        transformed_image = utils.window_image_for_holo(raytrace)
        holo = OSPR(transformed_image, False)
        file = '{}-holo.{}'.format(filename, 'bmp')
        cv2.imwrite('holograms/{}'.format(file), holo)
        print("Saved as", file)



