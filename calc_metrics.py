#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:03:52 2019

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
import pandas as pd
import numpy as np



class DataFormatError(Exception):
    """Raised when data not in correct format"""
    pass


def make_single_class(img, class_val):
    #print(class_val)
    out = img.copy()
    out[out != class_val] = 0
    out[out == class_val] = 1
    return np.array(out).astype(bool)
    
    
def calc_iou(gt, mask, smooth=0.0):
    intersection = np.sum(np.logical_and(gt, mask))
    union = np.sum(np.logical_or(gt, mask))
    iou = (intersection + smooth) / (union + smooth)
    
    return  iou
    
def calc_dice(gt, mask, smooth=0.0):
    intersection = np.sum(np.logical_and(gt, mask))
    masks_sum= gt.sum() + mask.sum()
    dice = ( 2 * (intersection + smooth))  / (masks_sum + smooth)
    
    return dice
    

gt_path = "/home/aries/ds_course/term3/bee_data_1024/gt_class_masks_1024"
unet_path = "/home/aries/ds_course/term3/bee_data_1024/unet_predictions_class_mask_1024"


class_values = {'Bee': 35, 'Abdomen': 100 }
NUM_CLASSES = 2


# match gt files with unet files
gt_paths = glob.glob(os.path.join(gt_path , "*.png"))
unet_paths = glob.glob(os.path.join(unet_path, "*.png"))
assert len(gt_paths) == len(unet_paths)
gt_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
unet_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
paired_masks = list(zip(gt_paths, unet_paths))


# dataframe for scores



all_scores = []


for gt_path, unet_path in paired_masks:
    if not os.path.basename(gt_path) == os.path.basename(unet_path):
        raise DataFormatError("Names of ground truth and predicted masks must match")
        

    gt_img = cv2.imread(gt_path) #, cv2.IMREAD_GRAYSCALE)
    unet_img = cv2.imread(unet_path) #, cv2.IMREAD_GRAYSCALE)
    
    if not gt_img.shape == unet_img.shape:
        raise DataFormatError("Images must be the same size")
    
    filename = os.path.basename(gt_path)
    metrics = [filename]
    
    dice_scores = []
    iou_scores = []
    
    #Bee, then Abdomen
    class_vals = sorted(list(class_values.values()))
    for val in class_vals:
        
        gt_class_mask = make_single_class(gt_img, val)
        unet_class_mask = make_single_class(unet_img, val)


        iou_score = calc_iou(gt_class_mask, unet_class_mask)
        dice_score = calc_dice(gt_class_mask, unet_class_mask)

        iou_scores.append(iou_score)
        dice_scores.append(dice_score)
        
    
    # get average scores
    avg_iou = np.average(iou_scores)
    avg_dice = np.average(dice_scores)

    # make csv entry
    metrics.extend(iou_scores)
    metrics.append(avg_iou)
    metrics.extend(dice_scores)
    metrics.append(avg_dice)
    #print(metrics)

    
    all_scores.append(metrics)
    
    
#print(np.array(all_scores))
    
df = pd.DataFrame(all_scores, columns=['Image', 'Bee Jaccard', 'Ab Jaccard', 'Avg Jaccard', 'Bee Dice', 'Ab Dice', 'Avg Dice'])
print(df.head())
df.to_csv("scores_RCNN_GT_12Sept.csv")