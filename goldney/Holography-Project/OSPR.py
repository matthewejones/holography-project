# Written for CMMPE by Adam Goldney
# Copyright 2020-2021

# All functions and libs contained in holo_projector.py
from holo_projector import *


adaptive = False
rotate = False
gamma = False
invert = False
scale = 1

# OSPR Algorithm
def OSPR(target_image, adaptive=False):
    A = np.sqrt(target_image)
    TotalHologram = np.zeros([HologramHeight, HologramWidth,3])
    RField = np.zeros([HologramHeight, HologramWidth])
    frame_no = 1
    
    for i in range(0,3):
        Hologram = np.zeros([HologramHeight, HologramWidth])

        for iteration in range(0, N):
            Diffuser = np.exp(1j * 2 * math.pi * np.random.rand(HologramHeight, HologramWidth))
            Efield = A * Diffuser
            Hfield = np.fft.ifft2(np.fft.ifftshift(Efield))
            Phase = np.angle(Hfield) > 0
            Hologram += Phase * 2**(iteration)
            
            if adaptive == True:
                I = abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * np.pi * Phase), norm='ortho')))**2
                #I[:512,:] = 0
                RField += I
                alphaSq = abs(np.sum(RField)/(frame_no * np.sum(target_image)))
                
                ASq = (frame_no)*target_image - RField/alphaSq
                
                ASq = np.uint16(ASq * np.uint16(ASq > 0))

                A = (np.sqrt(ASq) + np.sqrt(target_image)) / 2
                cv2.imshow("",(ASq).astype(np.uint16)/np.percentile(ASq,99))
                cv2.waitKey(0)
                frame_no += 1
                

        TotalHologram[:,:,i] = Hologram
    return TotalHologram



# Get the input image
argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv,"hr:agi",["rotate=","scale="])
except:
    raise TypeError("Enter valid input image. Type OSPR.py -h for usage")
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('Usage:\n\tpython OSPR.py <inputimage> (Contained in /Images/ folder) \nFlags: \n\t-r <angle> :\t\tRotate input image \n\t--rotate <angle> :\tRotate input image \n\t-a :\t\t\tAdaptive OSPR\n\t-g :\t\t\tEnable Gamma Correction\n\t-i :\t\t\tInvert Image,\n\t--scale <scale> :\t\tScale Image')
        sys.exit(2)
    elif opt in ("-r", "--rotate"):
        angle = float(arg)
        rotate = True
    elif opt == "-a":
        adaptive = True
    elif opt == "-g":
        gamma = True
    elif opt == "-i":
        invert = True
    elif opt == "--scale":
        scale = float(arg)
        

img = args[0]
img_name = img.partition(".")[0]
input_image = cv2.imread('Images/' + img, cv2.IMREAD_GRAYSCALE)


print("\n")
print("#" * 73)
print("#\tOSPR Hologram Generator for Freeman Projector\t\t\t#\n#\tAdam Goldney, Cambridge Engineering 4th Year Project\t\t#")
print("#" * 73)


if input_image is None:
    raise ValueError("No image with that name")


# Image inversion white -> black
if invert == True:
    input_image = 255 - input_image

# Rotate image
if rotate == True:
    input_image = rotate_image(input_image, angle)
    img_name = img_name + '-titled_{}'.format(str(int(angle)))

# Gamma correction
if gamma == True:
    input_image = gamma_correct(input_image)

# Scale image by desired amount
input_image = scale_image(input_image, scale)

# Transform for projection
transformed_image = window_image_for_holo(input_image)
    
# Create the hologram using OSPR
hologram = OSPR(transformed_image, adaptive)


# Create a directory if it doesn't exist
if not os.path.exists('Holograms'):
    os.makedirs('Holograms')


# Write the hologram to file
cv2.imwrite('Holograms/Holo-'+ img_name + '.bmp', hologram)

print('Holo-' + img_name + '.bmp')

