import  numpy as np
import  math
import  cv2

image_height = 192
image_width = 192
N = 8

for  number_of_frames  in  range (50):
    target_image = np.random.rand(image_height , image_width , 3)
    ref = np.exp(1j * np.ones([ image_height , image_width , 3]) * 2 * math.pi * np.random.rand())
    ref = ref / np.linalg.norm(ref)
    A = np.sqrt(target_image)
    Hologram = np.zeros ([ image_height , image_width , 3])
    for  iteration  in  range(0, N):
        Diffuser = np.exp(1j * 2 * math.pi * np.random.rand(image_height, image_width , 3))
        Efield = ref * A * Diffuser
        Hfield = np.fft.ifft2(np.fft.ifftshift(Efield))
        Phase = np.angle(Hfield) > 0
        Hologram  += Phase * 2**( iteration)
    Hologram  /= 255
    #cv2.imshow(’Hologram ’, Hologram)
    #cv2.waitKey (0)