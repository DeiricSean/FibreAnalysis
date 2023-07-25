# import numpy as np

# # Sample performance data for four models (replace with your actual data)
# model1_performance = [0.85, 0.78, 0.89, 0.91, 0.83]
# model2_performance = [0.82, 0.79, 0.85, 0.87, 0.81]
# model3_performance = [0.88, 0.82, 0.87, 0.89, 0.90]
# model4_performance = [0.76, 0.75, 0.80, 0.79, 0.82]

# # Combine the data into a 2D array
# data = np.column_stack((model1_performance, model2_performance, model3_performance, model4_performance))

# print(data)


# Display examples with bounding box and masks
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

fibre_images = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data\Prepared\Test\images'
fibre_masks = r'C:\Users\dezos\Documents\Fibres\FibreAnalysis\Data\Prepared\Test\masks'

img_files = os.listdir(fibre_images) # Get list of files in the directory 




dataHolder = []


for d in random.sample(img_files, 2):
    fullPath = os.path.join(fibre_images, d)
    mask_path = os.path.join(fibre_masks, f"mask{d[5:]}")  # Change filename from image_ to mask _ to retrieve mask
    print(fullPath)  # Print File name 
    print(mask_path)  # Print File name 
    ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    
#    results = YOLOmodel(fullPath)

    
    myResult = GetYOLOResult(YOLOmodel(fullPath))   # get area and count for YOLO    

    # Calculate IoU and Dice coefficient
    iou_value, diceCoef= calculate_iou(ground_truth_mask, myResult[-1]["overallMask"])
    
    # Prepare rows for dataframe later. See this link as to why to create a list first (much faster)  https://stackoverflow.com/questions/10715965/create-a-pandas-dataframe-by-appending-one-row-at-a-time
    dataHolder.append([d, iou_value, diceCoef])
    
    print("IoU:", iou_value)
    print("Dice Coefficient:", diceCoef)


completeDataFrame = pd.DataFrame(dataHolder, columns=['image', 'YOLOiou','YOLODice'] )
print(completeDataFrame)


# Perform Kruskal-Wallis test
statistic, p_value = kruskal(data['Model 1'], data['Model 2'], data['Model 3'], data['Model 4'])

# Output the results
print("Kruskal-Wallis test statistic:", statistic)
print("P-value:", p_value)

# Interpret the results (using the same alpha as before)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the groups.")



# Perform Dunn's test with Bonferroni adjustment  - need other datagrame with all but relevant columns dropped 
dunn_results = sp.posthoc_dunn(data, p_adjust='bonferroni')

# Output the Dunn's test results
print("Dunn's test p-values with Bonferroni adjustment:")
print(dunn_results)

# Additional analysis (optional):
# To obtain a pairwise comparison table with significant differences marked, you can use the following:
pairwise_comparison_table = (dunn_results < alpha).astype(int)
pairwise_comparison_table.index = data.columns
pairwise_comparison_table.columns = data.columns
print("\nPairwise comparison table with significant differences (1 means significant, 0 means not significant):")
print(pairwise_comparison_table)




























## Display the image with full markup 
#    res_plotted = results[0].plot()  # assuming only one image per run 
#    cv2.imshow("result", res_plotted)             #https://docs.ultralytics.com/modes/predict/#plotting-results
#    # Set the window position to (x, y) coordinates (adjust these values as needed)
#    cv2.moveWindow("result", 100, 100) # window is being displayed off screen for some reason 
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()    




    
    
    
# def  display_YOLO()   
    
#     #     # Process results list
#     for result in results:
#     #     boxes = result.boxes.data  # Boxes object for bbox outputs
#     #     masks = result.masks.data  # Masks object for segmentation masks outputs
#     #     keypoints = result.keypoints.data  # Keypoints object for pose outputs
#     #     probs = result.probs.data  # Class probabilities for classification outputs
#         image = cv2.imread(fullPath)
#         #print(masks)
        
        
#         overlay_combined = np.zeros_like(image)
        
#         for j, mask in enumerate(result.masks.data):
#         #   # Move the mask tensor to CPU if it's on a CUDA device
#             mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask

#         #   # Convert the mask to uint8 and resize it to match the image size
#             mask_np = cv2.resize((mask_np.astype( np.uint8) * 255), (image.shape[1], image.shape[0]))

#                 # Create a red transparency mask (instead of grayscale)
            
#     # Convert the grayscale mask to 3-channel (RGB) format with red color
#             #red_mask = np.stack((mask_np,) * 3, axis=-1)
#             red_mask = np.zeros_like(image)
#             red_mask[:, :, 2] = mask_np  # Set the red channel to the mask value
#         #   # Create a transparency mask
#             transparency_mask = np.stack((mask_np,) * 3, axis=-1)

#         #   # Apply the mask overlay to the original image
#             # overlay = cv2.addWeighted(image, 1, transparency_mask, 0.5, 0)
#             overlay_combined = cv2.addWeighted(overlay_combined, 1, red_mask,1, 0)

#         # Combine the overlay with the original image
#         result_image = cv2.addWeighted(image, 0.7, overlay_combined, 0.3, 0)

#                 #   # Display the result
#         cv2.imshow('name',result_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()





# print('finished')    
    
    #img = cv2.imread(d["file_name"])
    #visualizer = Visualizer(img[:, :, ::-1], metadata=fibre_metadata, scale=0.5)
    #out = visualizer.draw_dataset_dict(d)
    #cv2_imshow(out.get_image()[:, :, ::-1])