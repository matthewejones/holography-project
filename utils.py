import numpy as np
from numba import jit

from goldney.holo_projector import *

@jit
def replay_field(hologram):
    return abs(np.fft.fftshift(np.fft.fft2(hologram, norm="ortho")))

def split_filename(filename):
    partition = filename[::-1].partition('.')
    name = partition[2][::-1]
    extension = partition[0][::-1]
    return name, extension