import numpy as np
from numba import jit

from goldney.holo_projector import *

@jit
def replay_field(hologram):
    return abs(np.fft.fftshift(np.fft.fft2(hologram, norm="ortho")))