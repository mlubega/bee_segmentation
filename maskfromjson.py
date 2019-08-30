#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 13:54:19 2019

@author: aries
"""

import os
import logging
import json
import cv2 
import argparse 
import numpy as np
import matplotlib.pyplot as plt

IMG_DIM = 1025

#CLASS_COLORS = { "bee": (130, 232, 232), "bee abdomen": (129, 27, 27) } #rgb
CLASS_COLORS = { "bee": 35, "bee abdomen": 100 } # grayscale
def setLogger():
    
    logger = logging.getLogger('CreateMasks')
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    
    return logger




def main(data_file, mask_dest_file):
    


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
                mask = np.zeros((IMG_DIM,IMG_DIM), np.uint8) 

                
                # iteratively add shapes
                regions = img["regions"]
                logger.debug("Adding %i regions", len(regions))
                
                for region in regions:
                    shape = region["shape_attributes"]["name"]
                    bee_type = region["region_attributes"]["class"]
                    color = CLASS_COLORS[bee_type]
                    
                    
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
                    
                        cv2.ellipse(mask, (x,y), (rx, ry), angle, 0, 360, color, -1)
                        
                    elif shape == "polygon":
                        
                        vertices = zip(region["shape_attributes"]["all_points_x"],
                                       region["shape_attributes"]["all_points_y"])
                        
                        
                        # change from list of tuples to list of array
                        vertices = np.array(list(map(list, vertices)), np.int32)
    
                        cv2.fillPoly(mask, [vertices], color)
                    else:
                        print(shape)
                        raise TypeError(shape, "is not supported")
                
                
                #cv2.imshow('Mask',mask)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                plt.imshow(mask)
                plt.show()
                
                #cv2.imwrite(os.path.join(mask_dest_file, filename), mask)
                        
                
                
            
            
    finally:
        logger.handlers = []
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True,  help="path to VIA's JSON file", type=str)
    parser.add_argument("--dst_folder", required=True,  help="path to destination folder", type=str)
    args = parser.parse_args()

    main(args.json, args.dst_folder)
    