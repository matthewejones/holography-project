import numpy as np
from numba import jit
import cv2

from goldney.holo_projector import *

shades = ["grad1.png", "grad2.png", "cylgrad1.png", "cylgrad2.png"]


@jit
def replay_field(hologram):
    return abs(np.fft.fftshift(np.fft.fft2(hologram, norm="ortho")))

def split_filename(filename):
    partition = filename[::-1].partition('.')
    name = partition[2][::-1]
    extension = partition[0][::-1]
    return name, extension


def shade(image, i):
    shade = cv2.imread("shade/{}".format(shades[i - 1]), cv2.IMREAD_GRAYSCALE)
    return cv2.bitwise_and(image, shade)