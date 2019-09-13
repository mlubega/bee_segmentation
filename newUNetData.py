#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:55:25 2019

@author: aries
"""

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
import math
import re

IMG_DIM = 1025

OFFSET_X = 0
OFFSET_Y = 0
CLASS_NUM = { "bee": 1, "bee abdomen": 2 } 

def setLogger():
    
    logger = logging.getLogger('CreateMasks')
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    
    return logger




def main(data_file, dest_folder):
    


    try:
        logger = setLogger()
        
        with open(data_file, 'r') as f:
            bee_segmentations = json.load(f)
            logger.info("%i elements in json file", len(bee_segmentations))
            
            image_data = bee_segmentations.values()
            
            for img in image_data:
                filename = img["filename"]
                file_num = re.findall("\d+", os.path.splitext(filename)[0])[0]
                logger.debug("Processsing %s", file_num)
                
                new_file = os.path.join(dest_folder, file_num + ".txt")
                f = open(new_file, 'w+')
                
                
                # get center points of  each region
                regions = img["regions"]
                logger.debug("Adding %i regions", len(regions))
                
                for region in regions:
                    shape = region["shape_attributes"]["name"]
                    bee_type = region["region_attributes"]["class"]
                    class_num = CLASS_NUM[bee_type]
                    
                    x = 0
                    y = 0
                    theta = 0.0
                    
                    

                    if shape == "circle":
                        
                        x = int(region["shape_attributes"]["cx"])
                        y = int(region["shape_attributes"]["cy"])
                        
                        if (x >= 1024) or (y >= 1024):
                            continue
                        
                        
                        line = "{:<6d} {:<6d} {:<6d} {:<6d} {:<6d} {:<5f}\n".format(OFFSET_X, OFFSET_Y, class_num, x, y, theta)
                        f.write(line)
    
                    elif shape == "ellipse":
                        
                        x = int(region["shape_attributes"]["cx"])
                        y = int(region["shape_attributes"]["cy"])
                        theta = region["shape_attributes"]["theta"]
                        
                        if theta < 0: 
                            theta += (2 * math.pi)
                        
                        if (x >= 1024) or (y >= 1024):
                            continue
                        
                        
                        line = "{:<6d} {:<6d} {:<6d} {:<6d} {:<6d} {:<5f}\n".format(OFFSET_X, OFFSET_Y, class_num, x, y, theta)
                        f.write(line)

                        
                    elif shape == "polygon":
                        
                        vertices = zip(region["shape_attributes"]["all_points_x"],
                                       region["shape_attributes"]["all_points_y"])
                        
                        
                        # change from list of tuples to list of array
                        vertices = np.array(list(map(list, vertices)), np.int32)
                        
                        #create black image on which to draw polygon
                        mask_copy = np.zeros((IMG_DIM,IMG_DIM), np.uint8)
                        cv2.fillPoly(mask_copy, [vertices], 255)
                        
                        # convert the grayscale image to binary image
                        retval, thresh = cv2.threshold(mask_copy, 0, 255, cv2.THRESH_BINARY)

                        # find center of polygon
                        M = cv2.moments(thresh)
                        
                        # calculate x,y coordinate of center
                        x = int(M["m10"] / M["m00"])
                        y = int(M["m01"] / M["m00"])
                        
                        if (x >= 1024) or (y >= 1024):
                            continue
                        
                         
                        # put text and highlight the center
                        #cv2.circle(mask, (cX, cY), 5, 0, -1)
                        
                        #special case
                        if (filename == "10cropped.png") and (bee_type == "bee") :
                            bee_type = "bee abdomen"
                            class_num = CLASS_NUM[bee_type]
                            
                        line = "{:<6d} {:<6d} {:<6d} {:<6d} {:<6d} {:<5f}\n".format(OFFSET_X, OFFSET_Y, class_num, x, y, theta)
                        f.write(line)
                            
                        

                    else:
                        print(shape)
                        raise TypeError(shape, "is not supported")

                f.close()
                #cv2.imshow('Mask',mask)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                #plt.imshow(mask)
                #plt.show()
                
                #cv2.imwrite(os.path.join(mask_dest_file, filename), mask)
                        
                
                
            
            
    finally:
        logger.handlers = []
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True,  help="path to VIA's JSON file", type=str)
    parser.add_argument("--dst_folder", required=True,  help="path to destination folder", type=str)
    args = parser.parse_args()

    main(args.json, args.dst_folder)
    