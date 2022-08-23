#Importing Necessary Files
import cv2
import numpy as np
import numpy as np
import os
import cv2



# defining the crack detector function
   
# here weak_th and strong_th are thresholds for
# double thresholding step
def PCD(img, weak_th = None, strong_th = None):
    
    # Bilateral Filter
    img = cv2.bilateralFilter(img,15,80,80)
    
    # conversion of image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Particle Denoising
    img = cv2.fastNlMeansDenoising(img,10,10,7,21)

    # Noise reduction step
    #img = cv2.GaussianBlur(img, (5, 5), 1.6)
       
    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
      
    # Conversion of Cartesian coordinates to polar 
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
       
    # setting the minimum and maximum thresholds 
    # for double thresholding
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
      
    # getting the dimensions of the input image  
    height, width = img.shape
       
    # Looping through every pixel of the grayscale 
    # image
    for i_x in range(width):
        for i_y in range(height):
               
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
               
            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
              
            # top right (diagonal-1) direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
              
            # top left (diagonal-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
               
            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0
   
    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)              
    ids = np.zeros_like(img)
       
    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):
              
            grad_mag = mag[i_y, i_x]
              
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2
       
       
    # finally returning the magnitude of
    # gradients of edges
    return mag

#Shi tomasi

def shi_tomasi(image):

    #Converting to grayscale
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners
    corners_img = cv2.goodFeaturesToTrack(gray_img,1000,0.01,10)
    
    corners_img = np.int0(corners_img)

    for corners in corners_img:
       
        x,y = corners.ravel()
        #Circling the corners in green0
        cv2.circle(image,(x,y),3,[0,255,0],-1)

    return image


#Driver Code
#Paste the path with name and extension of the image 
# Creating a VideoCapture object to read the video
#cap = cv2.VideoCapture('Sample.mp4')
 
 
# Loop until the end of the video
#while (cap.isOpened()):
 
    # Capture frame-by-frame
  #  ret, frame = cap.read()
   # frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
    #                     interpolation = cv2.INTER_CUBIC)
frame= cv2.imread('Test1.jpg')    
# Display the resulting frame
cv2.imshow('Frame', frame)
crack_frame = PCD(frame)
cv2.imshow('Cracked',crack_frame)    
# Bilateral Filter
frame = cv2.bilateralFilter(frame,15,80,80)
    
    # conversion of image to grayscale
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Particle Denoising
frame = cv2.fastNlMeansDenoising(frame,10,10,7,21)
#ShiTomasi Feature
shi= shi_tomasi(frame)
cv2.imshow('Shi',shi)
cv2.waitKey(0)