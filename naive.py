import numpy as np
import cv2

def naive(target_image):
    TotalHologram = np.zeros([target_image.shape[0], target_image.shape[1] ,3])
    Hfield = np.fft.ifft2(np.fft.ifftshift(np.sqrt(target_image)))
    Phase = np.angle(Hfield) > 0
    
    for i in range(3):
        TotalHologram[:, :, i] = Phase * 255
    
    return TotalHologram

if __name__ == "__main__":
    filename = 'images/transformed_grid.jpg'
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    hologram = naive(image)
    cv2.imwrite('holograms/naive.bmp', hologram)