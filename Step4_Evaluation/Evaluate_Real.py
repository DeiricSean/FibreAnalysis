
#pip install scikit-posthocs

import os
import random
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp

def GetYOLOResult(results):
    
    # Create a list to store dictionaries with area and count for each result
    result_data = []
    
    for result in results:
        areas = []
        image_shape = result.orig_shape
        overall_mask = np.zeros((result.orig_shape[0], result.orig_shape[1]), dtype=np.uint8)
        for j, mask in enumerate(result.masks.data):
        #   # Move the mask tensor to CPU if it's on a CUDA device
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        #   # Convert the mask to uint8 and resize it to match the image size
#            mask_np = cv2.resize((mask_np.astype( np.uint8) * 255), (result.orig_shape[1], result.orig_shape[0]))
 
            # Resize to original image size, Using INTER_NEAREST ensures resized mask remains binary with values 0 and 1, preserving its original nature. 
            mask_np_resized = cv2.resize(mask_np, (result.orig_shape[1], result.orig_shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            
            area_covered = np.sum(mask_np_resized)
            areas.append(area_covered)
            
            # Combine the current mask with the overall mask using logical OR operation
            overall_mask = cv2.bitwise_or(overall_mask, mask_np_resized)

        result_data.append({'area': sum(areas), 'count': len(areas), 'overallMask': overall_mask})
        
    return result_data  # get count of fibres and area covered

def calculate_iou(groundTruth, predicted):
    intersection = np.logical_and(groundTruth, predicted).sum()
    union = np.logical_or(groundTruth, predicted).sum()
    iou = intersection / union
    
    dice_coefficient = 2 * np.sum(intersection) / (np.sum(groundTruth) + np.sum(predicted))
    return iou , dice_coefficient


preTrainedYOLO = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\YoloLTrain1.pt'
# Load a model
YOLOmodel = YOLO(preTrainedYOLO)  # pretrained YOLOv8n model

fibre_images = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data\Raw'
#fibre_masks = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data\Prepared\Test\masks'

img_files = os.listdir(fibre_images) # Get list of files in the directory 




dataHolder = []


for d in img_files:
    fullPath = os.path.join(fibre_images, d)
    #mask_path = os.path.join(fibre_masks, f"mask{d[5:]}")  # Change filename from image_ to mask _ to retrieve mask
    print(fullPath)  # Print File name 
    #print(mask_path)  # Print File name 
    #ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    
    results = YOLOmodel(fullPath)

        
    res_plotted = results[0].plot(labels=0, boxes=0) # assuming only one image per run 
    cv2.imshow("result", res_plotted)             #https://docs.ultralytics.com/modes/predict/#plotting-results
    # Set the window position to (x, y) coordinates (adjust these values as needed)
    cv2.moveWindow("result", 100, 100) # window is being displayed off screen for some reason 
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
