import utils
import numpy as np
from numba import jit
import cv2

filename = "mej.png"
image_name = filename.split('.')[0]
iterations = 1000

@jit
def dbs(image):
    #Start with random array of binary phase pixels (represented as 0 or 255)
    print(image)
    rng = np.random.default_rng()
    I0 = 255 * rng.integers(0, 2, (utils.HologramHeight, utils.HologramWidth))
    cv2.imwrite("file.bmp", I0)

    H0 = utils.replay_field(I0)

    print(I0.shape, H0.shape)

    C0 = np.sum(np.abs(image - H0))

    for i in range(iterations):
        I1 = I0
        x = np.random.randint(0, utils.HologramWidth - 1)
        y = np.random.randint(0, utils.HologramHeight - 1)
        val = I1[y][x]
        if val == 0:
            I1[y][x] = 255
        else:
            I1[y][x] = 0
        
        H1 = utils.replay_field(I1)
        C1 = np.sum(np.abs(image - H1))

        if C1 > C0:
            I0 = I1
            C0 = C1
    
    return I0
    


    



if __name__ == "__main__":
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = utils.window_image_for_holo(image)

    hologram = dbs(image)
    cv2.imwrite("holo_" + image_name + ".bmp", hologram)

    cv2.imshow('Generated Hologram', hologram)
    cv2.waitKey(0)


