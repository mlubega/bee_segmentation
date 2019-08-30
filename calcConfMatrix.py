#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 18:11:07 2019

@author: aries
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import logging
import json
import cv2 
import argparse 
import glob

import numpy as np



class DataFormatError(Exception):
    """Raised when data not in correct format"""
    pass

gt_path = "/home/aries/ds_course/term3/bee_data_1024/gt_class_masks_1024"
unet_path = "/home/aries/ds_course/term3/bee_data_1024/unet_predictions_class_mask_1024"

classes = {'Bee':0, 'Abdomen': 1, 'Background': 2}

addition = { 70:['Bee', 'Bee'],
             200: ['Abdomen', 'Abdomen'],
             0: ['Background', 'Background']}

subtraction = {-65: ['Bee', 'Abdomen'],
               35: ['Bee', 'Background'],
               65: ['Abdomen', 'Bee'],
               100: ['Abdomen', 'Background'],
               -35: ['Background', 'Bee'],
               -100: ['Background', 'Abdomen']}

NUM_CLASSES = 3

conf_matrices = []



gt_paths = glob.glob(os.path.join(gt_path , "*.png"))
unet_paths = glob.glob(os.path.join(unet_path, "*.png"))

assert len(gt_paths) == len(unet_paths)
    
gt_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
unet_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

paired_masks = list(zip(gt_paths, unet_paths))

for gt_path, unet_path in paired_masks:
    if not os.path.basename(gt_path) == os.path.basename(unet_path):
        raise DataFormatError("Names of ground truth and predicted masks must match")
        

    gt_img = cv2.imread(gt_path) #, cv2.IMREAD_GRAYSCALE)
    unet_img = cv2.imread(unet_path) #, cv2.IMREAD_GRAYSCALE)
    
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES)) 

    gt = gt_img.astype(int)
    unet = unet_img.astype(int)
    
    added = gt + unet
    subtracted = gt - unet
    
    a_vals, a_counts = np.unique(added, return_counts=True)
    s_vals, s_counts = np.unique(subtracted, return_counts=True)
    
    
    for k, v in addition.items():
        row, col = v
        idx = list(a_vals).index(k)
        count = a_counts[idx]
        conf_matrix[classes[row], classes[col]] += count
        
    for k, v in subtraction.items():
        row, col = v
        idx = list(s_vals).index(k)
        count = s_counts[idx]
        conf_matrix[classes[row], classes[col]] += count
        
    print(os.path.basename(gt_path))
    print(conf_matrix)
        
    conf_matrices.append(conf_matrix)
    
        

    #plt.imshow(gt_img)
    #plt.show
    
    #plt.imshow(unet_img)
    #plt.show
                 
    #cv2.imshow('GT',gt_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
                 
    #cv2.imshow('UNET',unet_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



#plt.imshow(unet_img)
#plt.show



