import numpy as np
import math

image_height = 1080
image_width = 1920
frames = 4
N = 8

def OSPR(target_image):
    A = np.sqrt(target_image)
    TotalHologram = np.zeros([image_height, image_width,3])
    
    for i in range(0,3):
        Hologram = np.zeros([image_height, image_width])

        for iteration in range(0, N):
            Diffuser = np.exp(1j * 2 * math.pi * np.random.rand(image_height, image_width))
            Efield = A * Diffuser
            Hfield = np.fft.ifft2(np.fft.ifftshift(Efield))
            Phase = np.angle(Hfield) > 0
            Hologram += Phase * 2**(iteration) #binary shift
            print(Hologram)
                

        TotalHologram[:,:,i] = Hologram
    return TotalHologram

for n in range(frames):
    image = np.random.rand(image_height, image_width)
    hologram = OSPR(image)