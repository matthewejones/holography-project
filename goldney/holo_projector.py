# Written for CMMPE by Adam Goldney
# Copyright 2020-2021

# Contains all accessory functions for hologram generation
import numpy as np
import math
import cv2
import sys, getopt, os

HologramHeight  = 1024
HologramWidth   = 1280
N               = 8
pitch           = 13.62e-6
wavelength      = 532e-9
beam_angle      = 0.131 * np.pi
#f               = 0.15              # Focal length of lens

# Slicing parameters
slice_type      = 'polar'
slice_radius    = 0.05
no_slices       = 5



# Parameters:
#     shape     : the ideal shape of input image, None for original size.
#     theta     : rotation around the x axis
#     phi       : rotation around the y axis
#     gamma     : rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image
# 
# Reference:
#     1.        : http://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
#     2.        : http://jepsonsblog.blogspot.tw/2012/11/rotation-in-3d-using-opencvs.html

class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, img):
        self.image = img
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        
        # Get radius of rotation along 3 axes
        [rtheta, rphi, rgamma] = [i * np.pi / 180.0 for i in [theta, phi, gamma]]
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)
        
        return cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))

    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):
        
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))
        

def rotate_image(img, angle):
    '''
    Iterate through corner angle range
    '''
    it = ImageTransformer(img)
    #for ang in angles:
        
#       rotating an image along y-axis with pixel shift in +X direction
    rotated_img = it.rotate_along_axis(phi = angle, dx = 0)
    
#       rotating an image along yz-axis from 0 to 360 degree
#       rotated_img = it.rotate_along_axis(phi = ang, gamma = ang)
    
#       rotating an image along z-axis(Normal 2D) from 0 to 360 degree
#       rotated_img = it.rotate_along_axis(gamma = ang)

    
    return rotated_img
   
   
# Scale image to desired size, returns image of the same dimension as original with black space filling when smaller, or cropping when larger
def scale_image(img, scale=1):

    if img.ndim > 2:
        ret_img = np.zeros((HologramHeight, HologramWidth, 3), np.uint8)
    else:
        ret_img = np.zeros((HologramHeight, HologramWidth), np.uint8)
        
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    
    
    if scale > 1:
        # Anchor scale at the top of the image
        ret_img = img[0:HologramHeight, int((img.shape[1] - HologramWidth)/2) : int((HologramWidth+img.shape[1])/2)]

    elif scale == 1:
        ret_img = img
    
    elif scale < 1:
        # Anchor scale at the top of the image
        ret_img[0:img.shape[0], int((HologramWidth-img.shape[1])/2) : int((HologramWidth+img.shape[1])/2)] = img
        
    return ret_img
    
    
# Window and flip image for projection
def window_image_for_holo(img):
    # Flip the image since we'll be projecting it
    img = np.flip(img,1)
    
    if img.shape[0] / (HologramHeight/2) > img.shape[1] / HologramWidth:
        shrink_percentage = HologramHeight / 2 / img.shape[0]
    else:
        shrink_percentage = HologramWidth / img.shape[1]

    img = cv2.resize(img, (int(img.shape[1] * shrink_percentage), int(img.shape[0] * shrink_percentage)))
    
    if img.ndim > 2:
        ret_img = np.zeros((HologramHeight, HologramWidth, 3), np.uint8)
    else:
        ret_img = np.zeros((HologramHeight, HologramWidth), np.uint8)
    
    ret_img[int(HologramHeight / 2) : int(HologramHeight / 2 + img.shape[0]), int((HologramWidth-img.shape[1])/2) : int((HologramWidth+img.shape[1])/2)] = img
    return ret_img
    

# Window and flip set of images for hologram
def window_images(imgs):
    
    transformed_image = np.zeros([np.shape(imgs)[0], HologramHeight, HologramWidth])
    
    for i in range(0,np.shape(imgs)[0]):
        transformed_image[i] = window_image_for_holo(imgs[i])
        
    return transformed_image


# Gamma Correction
def gamma_curve(x):
    # Values obtained from Jinze Sha 4th Year Project

    return 1-(1-x)**0.3


def gamma_correct(img):

    img_norm = img / 255
    
    for i in range(0, HologramHeight):
        for j in range(0, HologramWidth):
            img[i,j] = gamma_curve(img_norm[i,j]) * 255

    return(img)


# Slice images with basic geometery to create multiple Fresnel slices
def slice_image(img, z0, type, r, no_slices):
    
    
    # Resize the image to the hologram dimensions
    if img.shape[0] / (HologramHeight) > img.shape[1] / HologramWidth:
        shrink_percentage = HologramHeight / img.shape[0]
    else:
        shrink_percentage = HologramWidth / img.shape[1]

    img = cv2.resize(img, (int(img.shape[1] * shrink_percentage), int(img.shape[0] * shrink_percentage)))
    ret_img = np.zeros([HologramHeight, HologramWidth])
    
    ret_img[int((HologramHeight-img.shape[0])/2):HologramHeight-int((HologramHeight-img.shape[0])/2), int((HologramWidth-img.shape[1])/2):HologramWidth-int((HologramWidth-img.shape[1])/2)] = img


    # Calculate scaled pixel value for r
    r_pixel = r * HologramWidth / (2*z0*np.tan(beam_angle)) * 2     # Multiply by 2 since we're shrinking the hologram for projection

    bins = np.linspace(0, r_pixel, no_slices+1)     # Create specified number of bins in z range
    
    
    # Hemispherical sliced depth shell
    if type == 'polar':
        
        x, y = np.meshgrid(np.arange(0, HologramWidth), np.arange(0, HologramHeight))
        u = (x - (HologramWidth + 1) / 2)
        v = (y - (HologramHeight + 1) / 2)
        
        vmax = np.vectorize(max)
        z = np.sqrt(vmax(0, r_pixel**2 - u**2 - v**2))  # z0 plane at base of hemisphere
    
    elif type == 'left':
        w = np.shape(ret_img)[0]
        x,y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,w))
        z = depth * np.linspace(-0.5,0.5,np.shape(ret_img)[0])
        
    elif type == 'right':
        z = - depth * np.linspace(-0.5,0.5,np.shape(ret_img)[0])
    
    
    # Create slices using bins
    z_slices = np.digitize(z,bins) - 1

    zs = z0 + np.linspace(0, r, no_slices+1)[:-1]
    
    
    # Create blank array for image slices
    slices = np.zeros([no_slices, HologramHeight, HologramWidth])

    
    for i in range(0, HologramHeight):
        for j in range(0, HologramWidth):
            slices[z_slices[i,j],i,j] = ret_img[i,j]
            
    return slices, zs
    


