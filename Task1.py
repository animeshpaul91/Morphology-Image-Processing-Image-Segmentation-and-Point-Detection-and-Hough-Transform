#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:06:39 2018

@author: Animesh Paul
UB Person #: 50290441
References: 
    1. https://www.imageeprocessing.com/
    2. https://stackoverflow.com/
"""
import cv2 as cv
import numpy as np
import os
import time

UBIT = 'animeshp'
np.random.seed(sum([ord(c) for c in UBIT]))

RESULT_DIR = "./Results/Task1/"
image = "./Inputs/noise.png"

n = 5 #Order of Kernel
pad_factor = int((n-1)/2) #Number of Padding
k = np.int64(np.ones(n*n).reshape(n,n)) #Structuring Element for Dilation and Erosion

def imgread(img):
    img = cv.imread(img, 0)
    r, image = cv.threshold(img, 127, 255, cv.THRESH_BINARY) #Binary Image
    return (image)

def save_img(filename, img):
    # Saves image with passed filename in the results folder
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    filename = os.path.join(RESULT_DIR, filename)
    cv.imwrite(filename, img)
    
def pad_zero(r, c, img):
    padded = np.int64(np.zeros((r,c)))
    for i in range(0,len(img)):
        for j in range(0,len(img[i])): #Zero Padded Array with m+2 rows and n+2 column
            padded[i+pad_factor][j+pad_factor] = img[i][j] # Temp holds the complete padded array list.
    return padded

def pad_one(r, c, img):
    padded = np.int64(np.ones((r,c)))
    for i in range(0,len(img)):
        for j in range(0,len(img[i])): #Zero Padded Array with m+2 rows and n+2 column
            padded[i+pad_factor][j+pad_factor] = img[i][j] # Temp holds the complete padded array list.
    return padded

def dilate(img): #Padded Image and Kernel
    img = pad_zero(r + 2*pad_factor, c + 2*pad_factor, img) #Padded Image Matrix
    d = np.int64(np.zeros((r,c))) #d will store dilated image
    for i in range(len(img) - 2*pad_factor):
        row = np.arange(i,i+n)
        for j in range(len(img[i]) - 2*pad_factor):
            col = np.arange(j,j+n)
            part = img[row[:, None], col] #Sliced Array for Operation
            d[i][j] = np.amax(part & k)
            if d[i][j] == 1: 
                d[i][j] = 255
    return(d)

def erode(img): #Padded Image and Kernel
    img = pad_one(r + 2*pad_factor, c + 2*pad_factor, img) #Padded Image Matrix
    e = np.int64(np.zeros((r,c))) #d will store dilated image
    for i in range(len(img) - 2*pad_factor):
        row = np.arange(i,i+n)
        for j in range(len(img[i]) - 2*pad_factor):
            col = np.arange(j,j+n)
            part = img[row[:, None], col] #Sliced Array for Operation
            temp = part & k
            if np.array_equal(temp, k):
                e[i][j] = 255
            else:
                e[i][j] = 0
    return(e)

def open_img(img):
    img_eroded = erode(img) #Eroded Image
    img_opened = dilate(img_eroded)
    return (img_opened)

def close_img(img):
    img_dilated = dilate(img) #Dilated Image
    img_closed = erode(img_dilated)
    return (img_closed)

def algo1(img): #Algorithm1
    img_open = open_img(img) #Opened Image
    final_img = close_img(img_open) #Followed by Closing
    save_img('res_noise1.jpg', final_img)
    return (final_img)
    
def algo2(img): #Algorithm2
    img_close = close_img(img) #Opened Image
    final_img = open_img(img_close) #Followed by Closing
    save_img('res_noise2.jpg', final_img)
    return (final_img)

def boundary_extract(img):
    img_eroded = erode(img)
    bdr = np.subtract(img, img_eroded)
    return (bdr)
     
if __name__ == "__main__":
    start_time = time.time()
    img = imgread(image)
    r,c = img.shape
    img1 = algo1(img)
    img2 = algo2(img)
    #Boundary Extraction starts Here
    bdr1 = boundary_extract(img1)
    bdr2 = boundary_extract(img2)
    save_img('res_bound1.jpg', bdr1)
    save_img('res_bound2.jpg', bdr2)
    print("\n--- %s seconds ---" % (time.time() - start_time))