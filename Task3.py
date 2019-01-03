#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:02:02 2018

@author: Animesh Paul
UB Person #: 50290441
References:
    1. https://docs.opencv.org/
    2. https://www.stackoverflow.com/
"""
import cv2 as cv
import numpy as np
import os
import time

UBIT = 'animeshp'
np.random.seed(sum([ord(c) for c in UBIT]))

RESULT_DIR = "./Results/Task3/"
image = "./Inputs/hough.jpg"

def read_and_detect_edges(img):
    img_c = cv.imread(img) #Colored Image
    img_g = cv.imread(img, 0) #Grayscale Image
    edges = cv.Canny(img_g, 100, 200) #Edge Detected Image
    return (img_c, img_g, edges)

def save_img(filename, img):
    # Saves image with passed filename in the results folder
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    filename = os.path.join(RESULT_DIR, filename)
    cv.imwrite(filename, img)

def vote(img): #Edge Detected & Binarized Image
    A = np.zeros((2*d, 181), dtype=np.uint8) #Accumulator
    t = np.deg2rad(np.arange(-90, 91)) #Stores all theta values in radian
    cos = np.cos(t) #Cosine values
    sin = np.sin(t) #Sine values
    e = img > 0 #Boolean Filter edges in Numpy matrix
    y_ind, x_ind = np.nonzero(e) #Store the indices of edges
    
    #start Voting
    for i in range(0,len(x_ind)): #For all edges
        x = x_ind[i]
        y = y_ind[i]
        
        for t_ind in range(0,181): #Iterate over all theta from -90 to +90
            p = d + int(round(x * cos[t_ind] + y * sin[t_ind])) #d is added for every positive index
            A[p, t_ind] += 1
            
    return (A, sin, cos)

def draw_red_lines(img, A, sin=0, cos=1): #Retrieve x and y from hough space
    A = A[:, 90] #Slice A values for theta = 0 only (Vertical Lines)
    A = A > T #Filter values above a threshold only
    p = np.nonzero(A) #Return true indices
    p = p[0] #Convert into a 1D numpy array
    
    for i in range(0,len(p)):
        r = p[i] #Rho Value
        r = r - d #Retieve actual rho values as opposite to line number 54
        x0 = r * cos
        y0 = r * sin
        x1 = int(x0 + 1000*(-sin))
        y1 = int(y0 + 1000*(cos))
        x2 = int(x0 - 1000*(-sin))
        y2 = int(y0 - 1000*(cos))
        cv.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
        
    save_img('red_line.jpg', img)
    
def draw_blue_lines(img, A, sin, cos): #Retrieve x and y from hough space
    A = A[:, 55] #Slice A values for theta = -30 only
    A = A > T1 #Filter values above a threshold only
    p = np.nonzero(A) #Return true indices
    p = p[0] #Convert into a 1D numpy array
    
    for i in range(0,len(p)):
        r = p[i] #Rho Value
        r = r - d #Retieve actual rho values as opposite to line number 54
        x0 = r * cos
        y0 = r * sin
        x1 = int(x0 + 1000*(-sin))
        y1 = int(y0 + 1000*(cos))
        x2 = int(x0 - 1000*(-sin))
        y2 = int(y0 - 1000*(cos))
        cv.line(img, (x1,y1), (x2,y2), (255,0,0), 2)
        
    save_img('blue_lines.jpg', img)

def vote_for_circles(img): #Variable Radius
    A = np.zeros((2*d, 2*d, R-S), dtype=np.uint8) # 3D Accumulator for circles. Max Radius = 30
    t = np.deg2rad(np.arange(0, 361)) #Stores all theta values in radian
    cos = np.cos(t) #Cosine values
    sin = np.sin(t) #Sine values
    e = img > 0 #Boolean Filter edges in Numpy matrix
    y_ind, x_ind = np.nonzero(e) #Store the indices of edges
    #start Voting
    for i in range(0,len(x_ind)): #For all edges
        x = x_ind[i]
        y = y_ind[i] #At this point x,y are coordinates of a white pixel in image
        for rad in range(S,R): #Vary Radius from 42 to 49 A Third Dimension
            for t_ind in range(0,361): #Iterate over all theta from 0 to 360
                a = d + int(round(x - (rad * cos[t_ind]))) #Add d to positive indices
                b = d + int(round(y - (rad * sin[t_ind]))) #Add d to positive indices
                #print(a,b)
                A[a, b, rad-S] += 1
    return (A)

def draw_circles(img, A):
    A = A > T2 #Filter values above a threshold only
    p = np.nonzero(A) #Return true indices
    apoints = p[0] #Get all a points on abr space
    bpoints = p[1] #Get all b points on abr space
    rad = p[2] #Get all radius points on abr space
    
    for i in range(0,len(apoints)):
        a = apoints[i] #a point
        b = bpoints[i] #b point
        r = rad[i] #radius
        a = a - d #Retieve Actual a value
        b = b - d #Retieve Actual a value
        r = r + S #Retieve Actual radius
        cv.circle(img, (a,b), r, (255,0,0), 2)
    save_img('coin.jpg', img)
        
if __name__ == "__main__":
    start_time = time.time()
    img_c, img_g, edges  = read_and_detect_edges(image) #Colored, grayscale and edge detected image's numpy matrix
    r, c = img_g.shape
    d = int(round(np.sqrt(r**2 + c**2))) #Length of the diagonal. 
    T = 81 #Hough Threshold for Red lines
    T1 = 120 #Hough Threshold for Blue lines
    A, sin, cos = vote(edges) #Accumulator Obtained
    draw_red_lines(img_c, A)
    draw_blue_lines(img_c, A, sin[55], cos[55])
    
    #Bonus Task Starts here....................................................
    
    S = 20; R = 31 #Range of Radii considered
    T2 = 181 #Hough Threshold for Circles
    A = vote_for_circles(edges) #Accumulator obtained for drawing Circles
    draw_circles(img_c, A)
    print("\n--- %s seconds ---" % (time.time() - start_time))