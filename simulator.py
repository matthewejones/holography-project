import numpy as np
import utils
import cv2

filename = "holo_mej.bmp"

if __name__ == "__main__":
    hologram = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #replay = abs(np.fft.fft2(hologram, norm='ortho'))**2
    replay = utils.replay_field(hologram)
    cv2.imwrite('replay_'+ filename, replay)
