# Written for CMMPE by Adam Goldney
# Copyright 2020-2021

# All functions and libs contained in holo_projector.py
from holo_projector import *

adaptive = False
rotate = False
gamma = False
invert = False
maskTrue = False
sliced = False
scale = 1



# Calculate total holograms using OSPR
def FresnelOSPR(image,fv):

    # Calculate the Zernike Polynomial for the virtual lens
    x, y = np.meshgrid(np.arange(0, HologramWidth), np.arange(0, HologramHeight))
    u = pitch*(x - (HologramWidth + 1) / 2)
    v = pitch*(y - (HologramHeight + 1) / 2)
    
    ro  = np.sqrt(u**2+v**2)
    phi = np.tan(v/u)
    
    # Defocus aberration
    zern3 = ro**2
    
    x, y = np.meshgrid(np.linspace(-1,1,HologramWidth), np.linspace(-1,1,HologramHeight))
    
    ro  = np.sqrt(x**2+y**2)
    
    
    # Get the amplitude of the target image
    A = np.sqrt(image)
    TotalHologram = np.zeros([HologramHeight, HologramWidth,3])
    
    # Loop through 3 colour channels on Freeman Projector
    for i in range(0,3):
        Hologram = np.zeros([HologramHeight, HologramWidth])
        
        # Loop through 8 brightness levels in hologram for Freeman Projector
        for iteration in range(0, N):
            Total_Hfield = np.zeros([HologramHeight, HologramWidth],dtype=complex)
            
            # Calculate hologram for each slice
            for slice in range(0,image.shape[0]):
                
                # Add random phase
                Diffuser = np.exp(1j * 2 * math.pi * np.random.rand(HologramHeight, HologramWidth))
                Efield = A[slice] * Diffuser
                
                # Virtual lens
                L = np.exp(1j * np.pi * zern3 / (wavelength*fv[slice]))
                
                Hfield = np.fft.ifft2(np.fft.fftshift(Efield)) / L
                
                
                Total_Hfield += Hfield
            
            # Mask the hologram to create desired Bokeh effect
            Total_Hfield = Total_Hfield * mask + (1-mask) * Diffuser
            
            # Add binary phase to hologram
            Phase = np.angle(Total_Hfield) > 0
            Hologram += Phase * 2**(iteration)

        TotalHologram[:,:,i] = Hologram
    
    return TotalHologram
    
# Title print out
print("\n")
print("#" * 73)
print("#\tFresnel OSPR Hologram Generator for Freeman Projector\t\t#\n#\tAdam Goldney, Cambridge Engineering 4th Year Project\t\t#")
print("#" * 73, "\n")

# Get the input image(s)
argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv,"hr:agims",["rotate=","scale="])
except:
    raise TypeError("Enter valid input image. Use flag -h for usage")
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('#\tUsage:\n#\t\tpython FresnelOSPR.py <inputimage> OR <inputdirectory> (Contained in /Images/ folder) \n#\tFlags: \n#\t\t-r <angle> :\t\tRotate input image \n#\t\t--rotate <angle> :\tRotate input image \n#\t\t-a :\t\t\tAdaptive OSPR\n#\t\t-g :\t\t\tEnable Gamma Correction\n#\t\t-i :\t\t\tInvert Image\n#\t\t-m :\t\t\tAdd Bokeh mask,\n\t--scale <scale> :\t\tScale Image')
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
    elif opt == "-m":
        maskTrue = True
    elif opt == "-s":
        sliced = True
    elif opt == "--scale":
        scale = float(arg)

file_types = ('.png','.jpg',',jpeg','.bmp')
imgs = []

if args[0].endswith(file_types):
    imgdir = args[0].partition(".")[0]
    imgs.append(args[0])

else:
    imgdir = args[0]
    for file in os.listdir('Images/' + imgdir + '/'):
        if file.endswith(file_types):
            imgs.append(imgdir + '/' + file)
    imgs.sort()
    
f = float(input("\n#\tEnter distance from outer lens to Fourier Replay Field in meters: "))
# Get z values
print("\n#\tLens focal length = {:.2f}m \n#\t{} Images loaded, enter real world depths for each Fresnel slice in meters.".format(f,len(imgs)))

z = np.ones_like(imgs, dtype='float')
for i in range(0,len(imgs)):
    if len(imgs) > 1:
        z[i] = float(input("#\t" + imgs[i].partition("/")[2] + ": "))
    else:
        z[i] = float(input("#\t" + imgs[i] + ": "))
        
    z[i] = z[i]
zs = "-"
for slice in z:
    zs = zs + str(slice) + "_"
zs = zs[:-1]


# Mask for bokeh effect
if maskTrue == True:
    mask = cv2.imread('Images/butterfly_mask.bmp',cv2.IMREAD_GRAYSCALE) / 255
else:
    mask = np.ones([HologramHeight, HologramWidth])
    

# Import images into array with transforms
input_image = []


# Load input images and apply corrections
for i in range(0, len(imgs)):
    input_image.append(cv2.imread('Images/' + imgs[i], cv2.IMREAD_GRAYSCALE))
    
    if len(np.shape(input_image)) < 2:
        raise ValueError("No image or directory with that name")

    # Image inversion white -> black
    if invert == True:
        input_image[i] = 255 - input_image[i]

    # Rotate image
    if rotate == True:
        input_image[i] = rotate_image(input_image[i], angle)
        imgdir = imgdir + '-titled_{}'.format(str(int(angle)))

    # Gamma correction
    if gamma == True:
        input_image[i] = gamma_correct(input_image[i])
    
    # Scale image
    input_image[i] = scale_image(input_image[i], scale)


if sliced == True:

    # Slice 2D image for 3D object projection (current options: 'polar' - sphere of specified radiuus, 'left' and 'right' - linear left and right
    sliced_image, sliced_z = slice_image(input_image[0], z[0], slice_type, slice_radius, no_slices)
    
    if np.shape(input_image)[0] > 1:
        input_image = np.concatenate((input_image[1:], sliced_image), axis=0)
        z = np.concatenate((z[1:], sliced_z), axis=0)
    else:
        input_image = sliced_image
        z = sliced_z


# Window images for hologram generation
transformed_image = window_images(input_image)
    


# Convert real world depth to virtual depth plane
fv = np.ones_like(z)

for i in range(len(z)):
    if z[i] == f:
        fv[i] = 1e9
    else:
        fv[i] = f*z[i] / (f - z[i])
    
# Generate hologram using OSPR
hologram = FresnelOSPR(transformed_image, fv)


# Create a directory if it doesn't exist
if not os.path.exists('Holograms'):
    os.makedirs('Holograms')

# Write the hologram to file
cv2.imwrite('Holograms/Holo-'+ imgdir + zs +'.bmp', hologram)
print('#\n#\tOutput file:\tHolo-'+ imgdir + zs +'.bmp')


