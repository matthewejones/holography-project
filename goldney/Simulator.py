# Written for CMMPE by Adam Goldney
# Copyright 2020-2021

# All functions and libs contained in holo_projector.py
from holo_projector import *

fresnel = False
fv = 1e9                # If not Fresnel, ensure in far field
f = 0.3


# Run simulation
def simulate(holo, fv):

    # Calculate the Zernike Polynomial for the virtual lens
    x, y = np.meshgrid(np.arange(0, HologramWidth), np.arange(0, HologramHeight))

    u = pitch*(x - (HologramWidth + 1) / 2)
    v = pitch*(y - (HologramHeight + 1) / 2)
    zern3 = np.pi*(u**2 + v**2);

    replay = np.zeros([HologramHeight, HologramWidth])

    for iteration in range(0, N):
        Phase = (holo / 2**iteration) % 2
        
        # Virtual lens
        L = np.exp(1j * zern3 / (wavelength*fv))
        
        for i in range(0,3):
            replay += abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * np.pi * Phase[:,:,i]) * L, norm='ortho')))**2
            
    
    replay = np.flip(replay,1)
    
    replay = replay / np.percentile(replay,99.9) * 255
    
    return replay
    
    
# Get the input image
argv = sys.argv[1:]

# Default - don't display simulation
display = False

try:
    opts, args = getopt.getopt(argv,"hf:d",["fresnel="])
except:
    raise TypeError("Enter valid input hologram. Type OSPR.py -h for usage")
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('Usage:\n\tpython OSPR.py <input_hologram.bmp> (Contained in /Holograms/ folder) \nFlags: \n\t-f <z1,z2,...zN> :\t\tEvaluate at real depth z \n\t--fresnel <z1,z2,...zN> :\tEvaluate at real depth z \n\t-d : \t\tDisplay Simulation Output')
        sys.exit(2)
    elif opt in ("-f", "--fresnel"):
        # Update desired depth for Fresnel holograms
        z = float(arg)
        
        # Convert real world depth to virtual depth plane
        if z == f:
            fv = 1e9
        else:
            fv = f*z / (f - z)
        
        fresnel = True
                
    elif opt in ("-d"):
        display = True
        

img = args[0]

img_name = img.partition(".bmp")[0]
#img_name = img_name.partition("-")[2]

holo = cv2.imread('Holograms/' + img)

if holo is None:
    raise ValueError("No hologram with that name")

print("\n")
print("#" * 73)
print("#\tOSPR Hologram Simulator based on Freeman Projector\t\t#\n#\tAdam Goldney, Cambridge Engineering 4th Year Project\t\t#")
print("#" * 73)

# Simulation
replay = simulate(holo,fv)


# Create a directory if it doesn't exist
if not os.path.exists('Replay_Simulations'):
    os.makedirs('Replay_Simulations')


# Write the replay field to file
if fresnel == True:
    cv2.imwrite('Replay_Simulations/Replay-'+ img_name + "-" + str(z) +'.bmp', replay)
    
    if display == True:
        cv2.imshow('Simulated Replay Field at z = {:.0f}mm'.format(z*1000),replay/255)
        cv2.waitKey(0)
        
else:
    cv2.imwrite('Replay_Simulations/Replay-'+ img_name + '.bmp', replay)
    
    if display == True:
        cv2.imshow('Simulated Far Field Replay Field',replay/255)
        cv2.waitKey(0)


