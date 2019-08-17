#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:54:19 2019

@author: aries
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import logging
import json
import cv2 

import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

data_file = "/home/aries/ds_course/term3/bee_export.json"
mask_dest_file = "/home/aries/ds_course/term3/bee_data/masks_sm/"
IMG_DIM = 1025



CLASS_COLORS = { "bee": (130, 232, 232), "bee abdomen": (129, 27, 27) }

def setLogger():
    
    logger = logging.getLogger('CreateMasks')
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    
    return logger


def main():

    try:
        logger = setLogger()
        
        with open(data_file, 'r') as f:
            bee_segmentations = json.load(f)
            logger.info("%i elements in json file", len(bee_segmentations))
            
            image_data = bee_segmentations.values()
            
            for img in image_data:
                filename = img["filename"]
                logger.debug("Processsing %s", filename)
                
                #create black image on which to draw regions
                mask = np.zeros((IMG_DIM,IMG_DIM,3), np.uint8)
                
                # iteratively add shapes
                regions = img["regions"]
                logger.debug("Adding %i regions", len(regions))
                
                for region in regions:
                    shape = region["shape_attributes"]["name"]
                    bee_type = region["region_attributes"]["class"]
                    color = CLASS_COLORS[bee_type]
                    
                    logger.debug("Shape: %s, Class: %s", shape, bee_type)
                    
                    
                    #drawing functions
                    if shape == "circle":
                        x = int(region["shape_attributes"]["cx"])
                        y = int(region["shape_attributes"]["cy"])
                        radius = int(region["shape_attributes"]["r"])
    
                        cv2.circle(mask, (x,y), radius, color, -1 )
                    elif shape == "ellipse":
                        
                        x = int(region["shape_attributes"]["cx"])
                        y = int(region["shape_attributes"]["cy"])
                        rx = int(region["shape_attributes"]["rx"])
                        ry = int(region["shape_attributes"]["ry"])
                        theta = region["shape_attributes"]["theta"]
                        angle = np.degrees(theta)
                        logger.debug("Radians to Deg: %f to %f ----- args: %i, %i, %i, %i, %f, %s", theta, angle,x,y,rx,rx,angle,color)
                    
                        cv2.ellipse(mask, (x,y), (rx, ry), angle, 0, 360, color, -1 , shift = 3)
                        
                    elif shape == "polygon":
                        break;
                        vertices = zip(region["shape_attributes"]["all_points_x"],
                                       region["shape_attributes"]["all_points_y"])
                        
                        # change from list of tuples to list of array
                        vertices = np.array(list(map(list, vertices)), np.int32)
                        vertices = vertices.reshape((-1, 1, 2))
                        cv2.polylines(mask, [vertices], True, color, -1)
                        
                    else:
                        print(shape)
                        raise TypeError(shape, "is not supported")
                    
                plt.imshow(mask)
                plt.show()
                        
                
                
            
            
    finally:
        logger.handlers = []
        
main()
    