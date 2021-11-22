import pygame
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image
import numpy


def read_texture(filename):
    """
    Reads an image file and converts to a OpenGL-readable textID format
    """
    img = Image.open(filename)
    img_data = numpy.array(list(img.getdata()), numpy.int8)
    img_data = 255 - img_data
    textID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textID)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return textID


def main():
    pygame.init()
    display = (1280, 1024)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption('PyOpenGLobe')
    pygame.key.set_repeat(1, 10)    # allows press and hold of buttons
    gluPerspective(40, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)    # sets initial zoom so we can see globe
    glRotatef(90,1,0,0)
    glRotatef(135,0,0,1)
    lastPosX = 0
    lastPosY = 0
    texture = read_texture('Images/world.jpg')
    
    
        #glLight(GL_LIGHT0, GL_POSITION,  (0, 0, 1, 0)) # directional light from the front
    glLight(GL_LIGHT0, GL_POSITION,  (5, 5, 5, 1)) # point light from the left, top, front
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 0, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    
    
    glEnable(GL_DEPTH_TEST)

    while True:
        for event in pygame.event.get():    # user avtivities are called events

            # Exit cleanly if user quits window
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Rotation with arrow keys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    glRotatef(1, 0, 0, 1)
                if event.key == pygame.K_RIGHT:
                    glRotatef(1, 0, 0, -1)
                if event.key == pygame.K_UP:
                    glRotatef(1, 0, -1, 0)
                if event.key == pygame.K_DOWN:
                    glRotatef(1, 0, 1, 0)

            # Zoom in and out with mouse wheel
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # wheel rolled up
                    glScaled(1.05, 1.05, 1.05)
                if event.button == 5:  # wheel rolled down
                    glScaled(0.95, 0.95, 0.95)

            # Rotate with mouse click and drag
            if event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                dx = x - lastPosX
                dy = y - lastPosY
                mouseState = pygame.mouse.get_pressed()
                if mouseState[0]:

                    modelView = (GLfloat * 16)()
                    mvm = glGetFloatv(GL_MODELVIEW_MATRIX, modelView)

                    # To combine x-axis and y-axis rotation
                    temp = (GLfloat * 3)()
                    temp[0] = modelView[0]*dy + modelView[1]*dx
                    temp[1] = modelView[4]*dy + modelView[5]*dx
                    temp[2] = modelView[8]*dy + modelView[9]*dx
                    norm_xy = math.sqrt(temp[0]*temp[0] + temp[1]
                                        * temp[1] + temp[2]*temp[2])
                    glRotatef(math.sqrt(dx*dx+dy*dy),
                              temp[0]/norm_xy, temp[1]/norm_xy, temp[2]/norm_xy)

                lastPosX = x
                lastPosY = y

        # Creates Sphere and wraps texture
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE )

        qobj = gluNewQuadric()
        gluQuadricTexture(qobj, GL_TRUE)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        gluSphere(qobj, 1, 50, 50)
        gluDeleteQuadric(qobj)
        glDisable(GL_TEXTURE_2D)
        
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)


        # Displays pygame window
        pygame.display.flip()
        pygame.time.wait(10)
        
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (width, height), data)
    image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
    image.save('glutout.png', 'PNG')

main()
