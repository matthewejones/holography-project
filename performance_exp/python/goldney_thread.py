import numpy as np
import math
import threading

image_height = 1080
image_width = 1920
frames = 4
N = 8

class OSPR(threading.Thread):
    def run(self):
        target_image = np.random.rand(image_height, image_width)
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
                    

            TotalHologram[:,:,i] = Hologram
        return TotalHologram

thread_list = []
for number_of_frames in range(frames):
    thread_list.append(OSPR())
    thread_list[-1].start()
for each in thread_list:
    each.join()