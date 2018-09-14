#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:55:28 2018

@author: dhuruvapriyan
"""

import sys
import cv2
import os
import dlib
import imutils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import glob
from imutils import face_utils


def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
    	# of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    r,g,b=0,0,0
    i=0
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        if((r+g+b)<(color[0]+color[1]+color[2])):
            r,g,b=color[0]+10,color[1]+10,color[2]+10
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),color.astype("uint8").tolist(), -1)
        startX = endX
        i+=1
    print("skin colour is R:{} G:{} B:{}".format(r,g,b))    
    case_id=0
    case=['Very Light','Light','Mediterranean','Brown','Black']
#    vl=([1,244,242,245],[2,236,235,233],[3,250,249,247],[4,253,251,230],[5,243,246,230],[6,256,247,229],[7,250,240,239],[8,243,234,229],[9,244,241,234],[10,251,252,244],[11,252,248,237],[12,254,246,225],[13,255,249,225],[14,255,249,225],[15,241,231,195],[16,239,226,173],[17,224,210,147],[18,242,226,151],[19,235,214,159],[20,235,217,113],[21,227,196,103],[22,225,193,106],[23,223,193,123],[24,222,184,119],[25,199,164,100],[26,188,151,98],[27,156,107,67],[28,142,88,62],[29,121,77,48],[30,100,49,22],[31,101,48,32],[32,96,49,33],[33,87,50,41],[34,64,32,21],[35,49,37,41],[36,27,28,46])
    if(r>=225):
        case_id=1
    elif(r<225 and r>=170):
        case_id=2
    elif(r>170 and r>=130):
        case_id=3
    elif(r>130 and r<=90):
        case_id=4
    elif(r<90):
        case_id=5
    print("According to Von Luschan classification skin colour is :{}".format(case[case_id-1]))    
    return bar

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist
def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
win1 = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    image = cv2.imread(f)
    image = imutils.resize(image, width=500)
    image1=gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out_face = np.zeros_like(image)
    
    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
       """
       Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
       """
       shape = predictor(gray, rect)
       shape = face_utils.shape_to_np(shape)
    
       #initialize mask array
       remapped_shape = np.zeros_like(shape) 
       feature_mask = np.zeros((image.shape[0], image.shape[1]))   
    
       # we extract the face
       remapped_shape = face_remap(shape)
       cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
       feature_mask = feature_mask.astype(np.bool)
       out_face[feature_mask] = image1[feature_mask]
       win.set_image(out_face)
       clt = KMeans(n_clusters = 5)
       out_face = out_face.reshape((out_face.shape[0] * out_face.shape[1], 3))
       out_face=out_face[~np.all(out_face == 0, axis=1)]
#       print(np.max(out_face[:][0]),np.max(out_face[:][1]),np.max(out_face[:][2]))
       clt.fit(out_face)
       
       hist = centroid_histogram(clt)
       bar = plot_colors(hist, clt.cluster_centers_)  
        # show our color bart
       plt.figure()
       plt.axis("off")
       plt.imshow(bar)
       dlib.hit_enter_to_continue()   
       plt.pause(1)
       dlib.hit_enter_to_continue()
       plt.close('all')
     