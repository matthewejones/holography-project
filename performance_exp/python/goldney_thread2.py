import numpy as np
import math
import threading

image_height = 1080
image_width = 1920
frames = 4
N = 8


def sub_ospr(A, hologram, iteration, i):
    Diffuser = np.exp(1j * 2 * math.pi * np.random.rand(image_height, image_width))
    Efield = A * Diffuser
    Hfield = np.fft.ifft2(np.fft.ifftshift(Efield))
    Phase = np.angle(Hfield) > 0
    hologram[:, :, i] += Phase * 2**(iteration)

def OSPR(target_image):
    A = np.sqrt(target_image)
    TotalHologram = np.zeros([image_height, image_width,3])
    threads = []
    
    for i in range(0,3):
        for iteration in range(0, N):
            threads.append(threading.Thread(target=sub_ospr, args=(A, TotalHologram, iteration, i)))
            threads[-1].start()
    
    for thread in threads:
        thread.join()

    return TotalHologram

for n in range(frames):
    '''image = cv2.imread('C:/Users/matth/Documents/IIB/Project/holography-project/images/transformed_grid.jpg', cv2.IMREAD_GRAYSCALE)
    hologram = OSPR(image)
    cv2.imwrite('holo.bmp', hologram)'''
    image = np.random.rand(image_height, image_width)
    hologram = OSPR(image)