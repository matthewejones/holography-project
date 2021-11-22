from holo_projector import *

circle = np.zeros([HologramHeight, HologramWidth])

r0 = 44
r1 = 50


x, y = np.meshgrid(np.arange(0, HologramWidth), np.arange(0, HologramHeight))
u = (x - (HologramWidth + 1) / 2)
v = (y - (HologramHeight + 1) / 2)

for i in range(int(HologramHeight/2), HologramHeight):
    for j in range(0, HologramWidth):
        
        if r0**2 <= u[i,j]**2 + v[i,j]**2 <= r1**2:
            circle[i,j] = 255
            
            
cv2.imwrite('Images/half_32_ring.jpg', circle)
