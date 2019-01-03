#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 20:50:06 2018

@author: Animesh Paul
UB Person #: 50290441
References:
    1. https://docs.opencv.org/
    2. https://www.stackoverflow.com/
"""

import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import time

UBIT = 'animeshp'
np.random.seed(sum([ord(c) for c in UBIT]))

RESULT_DIR = "./Results/Task2/"
image1 = "./Inputs/point.jpg"
image2 = "./Inputs/segment.jpg"

def imgread(img):
    img = cv.imread(img, 0)
    return (img)

def save_img(filename, img):
    # Saves image with passed filename in the results folder
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    filename = os.path.join(RESULT_DIR, filename)
    cv.imwrite(filename, img)
    
def pad(r, c, img):
    padded = np.zeros((r,c))
    for i in range(0,len(img)):
        for j in range(0,len(img[i])): #Zero Padded Array with m+2 rows and n+2 column
            padded[i+pad_factor][j+pad_factor] = img[i][j] # Temp holds the complete padded array list.
    return padded

def run_mask(img): #Padded Image
    r = np.zeros((r1, c1)) #m stores the result after masking
    image = np.zeros((r1, c1)) #Stores the point detected image
    for i in range(len(img) - 2*pad_factor):
        row = np.arange(i,i+n)
        for j in range(len(img[i]) - 2*pad_factor):
            col = np.arange(j,j+n)
            part = img[row[:, None], col] #Sliced Array for Operation
            r[i][j] = abs(sum((mask*part).ravel()))
            #r[i][j] = sum((mask*part).ravel())
            if r[i][j] > T-1:
                image[i][j] = 255
                if j > 400 and j < 500 and i > 200 and i < 300:
                    print("\nCoordinates of Detected Point= "+str(i)+", "+str(j))
    return r, image

def segment(img): 
    #Plot Image Histogram
    x = np.arange(0,256)
    plt.hist(img.ravel(), x)
    plt.xlabel('Grayscale Values')
    plt.ylabel('Pixel Frequency')
    plt.title('Image Histogram')
    plt.savefig('./Results/Task2/Histogram.jpg')
    
    """ By observing the histogram it was empirically known that the highest gray values of the
    image lied in the range of 200 to 250. In order to segment the object from the background, 
    we need to choose a threshold value lesser than the values in this range lest we'll lose 
    parts of the object's grayscale values. We tried different values of threshold manually and
    found the best results at 198"""
    r, img = cv.threshold(img, 198, 255, cv.THRESH_BINARY) #Binary Thresholded Image
    save_img('segmented.jpg', img)
    
if __name__ == "__main__":
    start_time = time.time()
    mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) #Mask for Point Detection 
    #mask = np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,24,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]]) #Mask for Point Detection 
    n = mask.shape[0] #Order of Kernel
    pad_factor = int((n-1)/2) #Number of Padding
    T = 500 #Threshold
    img1 = imgread(image1) #Point
    img2 = imgread(image2) #Segment
    r1, c1 = img1.shape
    r2, c2 = img2.shape
    #img1 = pad(r1 + 2*pad_factor, c1 + 2*pad_factor, img1) #Padded Image Matrix
    R, img = run_mask(img1)
    save_img('points.jpg',img)
    #Task 2.1 Ends Here........................
    
    #Task 2.2 begins...........................
    c = segment(img2) #Coordinates of the Complete Bounding Box
    print("\n--- %s seconds ---" % (time.time() - start_time))